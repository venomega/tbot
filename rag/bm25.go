package main

import (
	"math"
	"sort"
	"strings"
)

const (
	defaultK1 = 1.5
	defaultB  = 0.75
)

type docFreq struct {
	Freqs      map[string]int
	TotalTerms int
}

func buildBM25(chunks []*chunkData, k1, b float64) BM25Params {
	n := len(chunks)
	if n == 0 {
		return BM25Params{K1: k1, B: b}
	}

	var totalTerms int
	df := make(map[string]int)
	for _, cd := range chunks {
		for term := range cd.Freqs {
			df[term]++
		}
		totalTerms += len(cd.Tokens)
	}

	avgdl := float64(totalTerms) / float64(n)
	idf := make(map[string]float64, len(df))
	for term, d := range df {
		idf[term] = math.Log(1 + (float64(n)-float64(d)+0.5)/(float64(d)+0.5))
	}

	return BM25Params{
		K1:          k1,
		B:           b,
		Avgdl:       avgdl,
		TotalChunks: n,
		IDF:         idf,
	}
}

// scoreChunkString is the original string-map-based scorer (used for legacy index fallback).
func scoreChunkString(queryTokens []string, freqs map[string]int, numTerms int, p BM25Params) float64 {
	var score float64
	for _, qt := range queryTokens {
		idf, ok := p.IDF[qt]
		if !ok {
			continue
		}
		f := float64(freqs[qt])
		docLen := float64(numTerms)
		score += idf * (f * (p.K1 + 1)) / (f + p.K1*(1-p.B+p.B*docLen/p.Avgdl))
	}
	return score
}

// scoreChunkPacked scores a chunk using packed term frequencies (termIdx<<16 | freq).
// queryTermIDs is a set (map) of query term numeric IDs that we care about.
// p.IDFValues must be populated (indexed by term ID).
func scoreChunkPacked(querySet map[uint32]struct{}, packed []uint32, numTokens int, p BM25Params) float64 {
	var score float64
	docLen := float64(numTokens)
	for _, entry := range packed {
		termIdx, freq := unpackFreq(entry)
		if _, ok := querySet[termIdx]; !ok {
			continue
		}
		idf := p.IDFValues[termIdx]
		ff := float64(freq)
		score += idf * (ff * (p.K1 + 1)) / (ff + p.K1*(1-p.B+p.B*docLen/p.Avgdl))
	}
	return score
}

func search(idx *Index, query string, topK int) []SearchResult {
	if len(idx.Chunks) == 0 {
		return nil
	}
	qtokens := tokenize(query)
	if len(qtokens) == 0 {
		return nil
	}

	// Pre-convert query tokens to term IDs for fast scoring (if we have TermIndex).
	// Also build a set for O(1) query-term lookup.
	var queryTermIDs []uint32
	querySet := make(map[uint32]struct{}, len(qtokens))
	if idx.TermIndex != nil {
		for _, t := range qtokens {
			if id, ok := idx.TermIndex[t]; ok {
				queryTermIDs = append(queryTermIDs, id)
				querySet[id] = struct{}{}
			}
		}
	}

	// Determine which chunks to score.
	// If we have an inverted index, only score chunks that contain at least one query term.
	// Otherwise fall back to scoring all chunks (legacy index).
	var candidateIndices []int
	if idx.Inverted != nil {
		seen := make(map[int]struct{})
		for _, qt := range qtokens {
			if indices, ok := idx.Inverted[qt]; ok {
				for _, ci := range indices {
					seen[ci] = struct{}{}
				}
			}
		}
		if len(seen) == 0 {
			return nil
		}
		candidateIndices = make([]int, 0, len(seen))
		for ci := range seen {
			candidateIndices = append(candidateIndices, ci)
		}
	} else {
		candidateIndices = make([]int, len(idx.Chunks))
		for i := range idx.Chunks {
			candidateIndices[i] = i
		}
	}

	type scored struct {
		chunk *Chunk
		score float64
	}

	pre := make([]scored, 0, len(candidateIndices))
	for _, ci := range candidateIndices {
		c := idx.Chunks[ci]
		var s float64

		if c.PackedFreqs != nil && len(querySet) > 0 && len(idx.BM25Params.IDFValues) > 0 {
			// Fast path: packed frequencies with term IDs
			s = scoreChunkPacked(querySet, c.PackedFreqs, c.NumTokens, idx.BM25Params)
		} else if c.Freqs != nil {
			// Fallback: pre-computed string-keyed frequencies (v3 compat)
			s = scoreChunkString(qtokens, c.Freqs, c.NumTokens, idx.BM25Params)
		} else {
			// Legacy fallback: tokenize on the fly
			freqs := countTokens(tokenize(c.Text))
			numTokens := 0
			for _, v := range freqs {
				numTokens += v
			}
			s = scoreChunkString(qtokens, freqs, numTokens, idx.BM25Params)
		}
		if s > 0 {
			pre = append(pre, scored{chunk: c, score: s})
		}
	}

	sort.Slice(pre, func(i, j int) bool {
		return pre[i].score > pre[j].score
	})

	if len(pre) > topK {
		pre = pre[:topK]
	}

	results := make([]SearchResult, 0, len(pre))
	for _, s := range pre {
		snippet := extractSnippet(s.chunk.Text, qtokens, 5)
		results = append(results, SearchResult{
			Path:    s.chunk.Path,
			Start:   s.chunk.Start,
			End:     s.chunk.End,
			Type:    s.chunk.Type,
			Name:    s.chunk.Name,
			Parent:  s.chunk.Parent,
			Imports: s.chunk.Imports,
			Symbols: s.chunk.Symbols,
			Lang:    s.chunk.Lang,
			Score:   s.score,
			Text:    s.chunk.Text,
			Snippet: snippet,
		})
	}
	return results
}

// extractSnippet finds the best matching region in text and returns
// up to `contextLines` lines of context around it.
func extractSnippet(text string, queryTokens []string, contextLines int) string {
	lines := strings.Split(text, "\n")
	if len(lines) == 0 {
		return ""
	}

	// Score each line by how many query tokens it contains
	lineScores := make([]int, len(lines))
	maxScore := 0
	bestLine := 0
	for i, line := range lines {
		lower := strings.ToLower(line)
		score := 0
		for _, qt := range queryTokens {
			if strings.Contains(lower, qt) {
				score++
			}
		}
		lineScores[i] = score
		if score > maxScore {
			maxScore = score
			bestLine = i
		}
	}

	if maxScore == 0 {
		// No direct match found, return the first lines
		bestLine = 0
	}

	// Extract context around the best line
	start := bestLine - contextLines
	if start < 0 {
		start = 0
	}
	end := bestLine + contextLines + 1
	if end > len(lines) {
		end = len(lines)
	}

	snippet := strings.Join(lines[start:end], "\n")
	return strings.TrimSpace(snippet)
}
