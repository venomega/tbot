package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

func main() {
	defer stopPythonChunker()
	if len(os.Args) < 2 {
		usage()
		return
	}
	cmd := os.Args[1]

	switch cmd {
	case "index":
		path := "."
		if len(os.Args) > 2 {
			path = os.Args[2]
		}
		abs, err := filepath.Abs(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		info, err := os.Stat(abs)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %s: %v\n", abs, err)
			os.Exit(1)
		}
		if !info.IsDir() {
			fmt.Fprintf(os.Stderr, "error: %s is not a directory\n", abs)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "Indexing %s...\n", abs)
		idx, err := indexDir(abs)
		if err != nil {
			fmt.Fprintf(os.Stderr, "index error: %v\n", err)
			os.Exit(1)
		}
		idx.RootPath = abs
		if err := saveIndex(idx); err != nil {
			fmt.Fprintf(os.Stderr, "save error: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "Index saved to %s\n", indexPath())
		out := map[string]interface{}{
			"ok":     true,
			"files":  len(idx.Files),
			"chunks": len(idx.Chunks),
		}
		data, _ := json.Marshal(out)
		fmt.Println(string(data))

	case "search":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: rag search <query> [top_k]")
			os.Exit(1)
		}
		query := os.Args[2]
		topK := 5
		if len(os.Args) > 3 {
			if v, err := strconv.Atoi(os.Args[3]); err == nil && v > 0 {
				topK = v
			}
		}
		if !indexExists() {
			// Auto-index before search
			fmt.Fprintf(os.Stderr, "No index found, indexing current directory...\n")
			idx, err := indexDir(".")
			if err != nil {
				fmt.Fprintf(os.Stderr, "index error: %v\n", err)
				os.Exit(1)
			}
			if err := saveIndex(idx); err != nil {
				fmt.Fprintf(os.Stderr, "save error: %v\n", err)
				os.Exit(1)
			}
		}
		idx, err := loadIndex()
		if err != nil {
			fmt.Fprintf(os.Stderr, "load index: %v\n", err)
			os.Exit(1)
		}
		result, err := searchIndex(idx, query, topK)
		if err != nil {
			fmt.Fprintf(os.Stderr, "search error: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(result)

	case "status":
		if !indexExists() {
			fmt.Println(`{"exists": false, "chunks": 0}`)
			return
		}
		idx, err := loadIndex()
		if err != nil {
			fmt.Fprintf(os.Stderr, "load index: %v\n", err)
			os.Exit(1)
		}
		// Count unique files
		files := make(map[string]bool)
		for _, c := range idx.Chunks {
			files[c.Path] = true
		}
		status := map[string]interface{}{
			"exists":      true,
			"chunks":      len(idx.Chunks),
			"files":       len(files),
			"avgdl":       idx.BM25Params.Avgdl,
			"total_terms": len(idx.BM25Params.IDF),
			"version":     idx.Version,
			"root_path":   idx.RootPath,
		}
		data, _ := json.Marshal(status)
		fmt.Println(string(data))

	default:
		usage()
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `rag — fast local RAG (no LLM)

Usage:
  rag index [path]     Index files in path (default: current dir)
  rag search <query>   Search indexed files, returns JSON
  rag status           Show index stats

The index is stored in ~/.config/tbot/rag_index/
`)
}
