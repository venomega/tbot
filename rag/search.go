package main

import (
	"encoding/json"
	"fmt"
)

func searchIndex(idx *Index, query string, topK int) (string, error) {
	if idx == nil || len(idx.Chunks) == 0 {
		return "[]", nil
	}
	results := search(idx, query, topK)
	if results == nil {
		return "[]", nil
	}
	data, err := json.Marshal(results)
	if err != nil {
		return "", fmt.Errorf("json: %w", err)
	}
	return string(data), nil
}
