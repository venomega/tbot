package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

func shouldSkip(name string) bool {
	skipDirs := map[string]bool{
		".git": true, ".svn": true, ".hg": true, "node_modules": true,
		"vendor": true, ".venv": true, "venv": true, "__pycache__": true,
		".opencode": true, ".terraform": true, "dist": true, "build": true,
		".next": true, ".nuxt": true, ".cache": true, ".ruff_cache": true,
	}
	if skipDirs[name] {
		return true
	}
	if strings.HasPrefix(name, ".") && name != "." {
		return true
	}
	return false
}

func indexDir(root string) (*Index, error) {
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("abs path: %w", err)
	}
	stat, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", root, err)
	}
	if !stat.IsDir() {
		return nil, fmt.Errorf("%s is not a directory", root)
	}

	var files []string
	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			if shouldSkip(info.Name()) && path != root {
				return filepath.SkipDir
			}
			return nil
		}
		if info.Size() == 0 || info.Size() > 500*1024 {
			return nil
		}
		files = append(files, path)
		return nil
	})

	numWorkers := runtime.NumCPU()
	if numWorkers < 1 {
		numWorkers = 1
	}

	type result struct {
		path   string
		chunks []*Chunk
		err    error
	}

	work := make(chan string, len(files))
	results := make(chan result, len(files))
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range work {
				chunks, err := chunkFile(path)
				results <- result{path: path, chunks: chunks, err: err}
			}
		}()
	}

	for _, f := range files {
		work <- f
	}
	close(work)
	wg.Wait()
	close(results)

	var allChunks []*chunkData
	fileMTimes := make(map[string]int64)
	totalFiles := 0
	totalErrors := 0

	for r := range results {
		if r.err != nil {
			totalErrors++
			continue
		}
		if len(r.chunks) == 0 {
			continue
		}
		totalFiles++
		rel, _ := filepath.Rel(root, r.path)
		for _, c := range r.chunks {
			if rel != "" {
				c.Path = rel
			}
			tokens := tokenize(c.Text)
			freqs := countTokens(tokens)
			allChunks = append(allChunks, &chunkData{
				Chunk:  c,
				Tokens: tokens,
				Freqs:  freqs,
			})
		}
		info, err := os.Stat(r.path)
		if err == nil {
			fileMTimes[rel] = info.ModTime().Unix()
		}
	}

	bm25Params := buildBM25(allChunks, defaultK1, defaultB)

	// Build a global term → numeric index mapping for compact storage,
	// and an IDFValues array for O(1) lookup during scoring.
	termIdx := uint32(0)
	numTerms := len(bm25Params.IDF)
	termIndex := make(map[string]uint32, numTerms)
	idfValues := make([]float64, numTerms)
	for term := range bm25Params.IDF {
		termIndex[term] = termIdx
		idfValues[termIdx] = bm25Params.IDF[term]
		termIdx++
	}
	bm25Params.IDFValues = idfValues

	chunks := make([]*Chunk, len(allChunks))
	inverted := make(map[string][]int)
	for i, cd := range allChunks {
		// Pack term frequencies into compact uint32 slices
		packed := make([]uint32, 0, len(cd.Freqs))
		for term, freq := range cd.Freqs {
			tidx := termIndex[term]
			if freq > maxPackedFreq {
				freq = maxPackedFreq // clamp, extremely rare
			}
			packed = append(packed, packFreq(tidx, uint32(freq)))
			// Build inverted index
			inverted[term] = append(inverted[term], i)
		}
		cd.Chunk.PackedFreqs = packed
		cd.Chunk.NumTokens = len(cd.Tokens)
		chunks[i] = cd.Chunk
	}

	idx := &Index{
		Version:    3,
		Chunks:     chunks,
		BM25Params: bm25Params,
		Files:      fileMTimes,
		TermIndex:  termIndex,
		Inverted:   inverted,
	}

	fmt.Fprintf(os.Stderr, "  Indexed %d files → %d chunks (%d errors)\n", totalFiles, len(chunks), totalErrors)
	return idx, nil
}
