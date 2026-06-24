package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

// Current index version. Increment when tokenization/index format changes
// so stale indices are automatically rebuilt.
const indexVersion = 4

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

// indexDir indexes a directory, with incremental support.
// If an existing index is found for the same root, only changed files are re-indexed.
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

	// ── Paso 1: Escanear árbol de archivos ──
	type fileInfo struct {
		path  string
		mtime int64
	}
	var currentFiles []fileInfo
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
		currentFiles = append(currentFiles, fileInfo{
			path:  path,
			mtime: info.ModTime().Unix(),
		})
		return nil
	})

	// ── Paso 2: Cargar índice existente (si hay) para diff incremental ──
	var existingIdx *Index
	existingFileMTimes := make(map[string]int64)

	if indexExists() {
		existingIdx, err = loadIndex()
		if err == nil && existingIdx.RootPath == root && existingIdx.Version == indexVersion {
			existingFileMTimes = existingIdx.Files
		} else {
			existingIdx = nil // incompatible or error, rebuild from scratch
		}
	}

	// ── Paso 3: Determinar qué archivos procesar ──
	type workItem struct {
		path  string
		rel   string
		mtime int64
	}

	// Pre-computar rel paths para búsqueda rápida
	currentRelPaths := make(map[string]int64, len(currentFiles))
	for _, f := range currentFiles {
		rel, _ := filepath.Rel(root, f.path)
		currentRelPaths[rel] = f.mtime
	}

	var toProcess []workItem
	var unchangedRelPaths []string // archivos que podemos reusar del índice anterior

	if existingIdx != nil {
		for _, f := range currentFiles {
			rel, _ := filepath.Rel(root, f.path)
			oldMtime, exists := existingFileMTimes[rel]
			if exists && oldMtime == f.mtime {
				unchangedRelPaths = append(unchangedRelPaths, rel)
			} else {
				toProcess = append(toProcess, workItem{
					path:  f.path,
					rel:   rel,
					mtime: f.mtime,
				})
			}
		}
		// Detectar archivos eliminados (log solo)
		for rel := range existingFileMTimes {
			if _, stillExists := currentRelPaths[rel]; !stillExists {
				fmt.Fprintf(os.Stderr, "  [incremental] %s deleted, will be removed\n", rel)
			}
		}
	} else {
		for _, f := range currentFiles {
			rel, _ := filepath.Rel(root, f.path)
			toProcess = append(toProcess, workItem{
				path:  f.path,
				rel:   rel,
				mtime: f.mtime,
			})
		}
	}

	// ── Paso 4: Chunkear y tokenizar archivos nuevos/modificados ──
	numWorkers := runtime.NumCPU()
	if numWorkers < 1 {
		numWorkers = 1
	}

	type chunkResult struct {
		rel    string
		chunks []*Chunk
		err    error
	}

	work := make(chan workItem, len(toProcess))
	results := make(chan chunkResult, len(toProcess))
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range work {
				chunks, err := chunkFile(item.path)
				if err == nil && len(chunks) > 0 {
					// Set relative paths
					for _, c := range chunks {
						c.Path = item.rel
					}
				}
				results <- chunkResult{rel: item.rel, chunks: chunks, err: err}
			}
		}()
	}

	for _, item := range toProcess {
		work <- item
	}
	close(work)
	wg.Wait()
	close(results)

	// ── Paso 5: Ensamblar todos los chunks ──
	var allChunks []*chunkData
	fileMTimes := make(map[string]int64)
	totalNewFiles := 0
	totalErrors := 0

	// 5a: Chunks nuevos/modificados
	for r := range results {
		if r.err != nil {
			totalErrors++
			continue
		}
		if len(r.chunks) == 0 {
			continue
		}
		totalNewFiles++
		for _, c := range r.chunks {
			tokens := tokenize(c.Text)
			freqs := countTokens(tokens)
			allChunks = append(allChunks, &chunkData{
				Chunk:  c,
				Tokens: tokens,
				Freqs:  freqs,
			})
		}
		// Buscar el mtime original
		for _, item := range toProcess {
			if item.rel == r.rel {
				fileMTimes[r.rel] = item.mtime
				break
			}
		}
	}

	// 5b: Chunks sin cambios (reusados del índice anterior)
	// Como el tokenizer cambió (versión 4), necesitamos re-tokenizar los chunks
	// existentes. Esto es mucho más barato que re-chunkear (solo O(texto) vs. AST).
	if existingIdx != nil && len(unchangedRelPaths) > 0 {
		// Construir mapa {relPath → []*Chunk} para lookup O(1)
		chunksByRel := make(map[string][]*Chunk)
		for _, c := range existingIdx.Chunks {
			chunksByRel[c.Path] = append(chunksByRel[c.Path], c)
		}

		for _, rel := range unchangedRelPaths {
			mtime, ok := existingFileMTimes[rel]
			if !ok {
				continue
			}
			for _, c := range chunksByRel[rel] {
				tokens := tokenize(c.Text)
				freqs := countTokens(tokens)
				allChunks = append(allChunks, &chunkData{
					Chunk:  c,
					Tokens: tokens,
					Freqs:  freqs,
				})
			}
			fileMTimes[rel] = mtime
		}
	}

	if len(allChunks) == 0 {
		return nil, fmt.Errorf("no files indexed (0 chunks)")
	}

	// ── Paso 6: Construir BM25 params ──
	bm25Params := buildBM25(allChunks, defaultK1, defaultB)

	// ── Paso 7: Construir term index e inverted index ──
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
		packed := make([]uint32, 0, len(cd.Freqs))
		for term, freq := range cd.Freqs {
			tidx := termIndex[term]
			if freq > maxPackedFreq {
				freq = maxPackedFreq
			}
			packed = append(packed, packFreq(tidx, uint32(freq)))
			inverted[term] = append(inverted[term], i)
		}
		cd.Chunk.PackedFreqs = packed
		cd.Chunk.NumTokens = len(cd.Tokens)
		chunks[i] = cd.Chunk
	}

	idx := &Index{
		Version:    indexVersion,
		RootPath:   root,
		Chunks:     chunks,
		BM25Params: bm25Params,
		Files:      fileMTimes,
		TermIndex:  termIndex,
		Inverted:   inverted,
	}

	totalFiles := totalNewFiles + len(unchangedRelPaths)
	fmt.Fprintf(os.Stderr, "  Indexed %d files → %d chunks (%d new, %d cached, %d errors)\n",
		totalFiles, len(chunks), totalNewFiles, len(unchangedRelPaths), totalErrors)
	return idx, nil
}
