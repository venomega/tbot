package main

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

var ragDir string

func init() {
	home, err := os.UserHomeDir()
	if err == nil {
		ragDir = filepath.Join(home, ".config", "tbot", "rag_index")
	}
}

func indexPath() string {
	return filepath.Join(ragDir, "index.gob")
}

func jsonPath() string {
	return filepath.Join(ragDir, "index.json")
}

func saveIndex(idx *Index) error {
	if err := os.MkdirAll(ragDir, 0755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}

	f, err := os.Create(indexPath())
	if err != nil {
		return fmt.Errorf("create gob: %w", err)
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	if err := enc.Encode(idx); err != nil {
		return fmt.Errorf("gob encode: %w", err)
	}

	// Also save as JSON for debugging
	jf, err := os.Create(jsonPath())
	if err != nil {
		return nil // non-fatal
	}
	defer jf.Close()
	encJ := json.NewEncoder(jf)
	encJ.SetIndent("", "  ")
	encJ.Encode(idx)

	return nil
}

func loadIndex() (*Index, error) {
	f, err := os.Open(indexPath())
	if err != nil {
		return nil, fmt.Errorf("open gob: %w", err)
	}
	defer f.Close()

	dec := gob.NewDecoder(f)
	var idx Index
	if err := dec.Decode(&idx); err != nil {
		return nil, fmt.Errorf("gob decode: %w", err)
	}
	return &idx, nil
}

func indexExists() bool {
	if ragDir == "" {
		return false
	}
	_, err := os.Stat(indexPath())
	return err == nil
}
