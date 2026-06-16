package main

import (
	"math"
	"sort"
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

func scoreChunk(queryTokens []string, freqs map[string]int, numTerms int, p BM25Params) float64 {
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

func search(idx *Index, query string, topK int) []SearchResult {
	if len(idx.Chunks) == 0 {
		return nil
	}
	qtokens := tokenize(query)
	if len(qtokens) == 0 {
		return nil
	}

	type scored struct {
		chunk     *Chunk
		freqs     map[string]int
		numTokens int
		score     float64
	}

	// Precompute precomputed data per chunk
	pre := make([]scored, 0, len(idx.Chunks))
	for _, c := range idx.Chunks {
		freqs := countTokens(tokenize(c.Text))
		numTokens := 0
		for _, v := range freqs {
			numTokens += v
		}
		s := scoreChunk(qtokens, freqs, numTokens, idx.BM25Params)
		if s > 0 {
			pre = append(pre, scored{chunk: c, freqs: freqs, numTokens: numTokens, score: s})
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
		})
	}
	return results
}
