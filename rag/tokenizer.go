package main

import (
	"strings"
	"unicode"
)

// Spanish stemmer: common suffixes to strip for basic normalization.
// This is a lightweight approach — not a full Snowball stemmer, but
// significantly reduces inflectional variance for Spanish text.
var spanishSuffixes = []string{
	"ándoles", "ándonos", "ándoos", "ándose", "ándole", "ándola",
	"ándolo", "ársela", "árselo", "érsela", "érselo", "írsela", "írselo",
	"ábamos", "íamos", "erais", "asteis", "isteis", "abais",
	"arían", "erían", "irían", "arías", "erías", "irías",
	"aréis", "eréis", "iréis", "arán", "erán", "irán",
	"arás", "erás", "irás", "aría", "ería", "iría",
	"ando", "endo", "iendo",
	"anza", "anzas", "ario", "arios", "aria", "arias",
	"ador", "adora", "adores", "adoras",
	"ante", "antes", "ente", "entes",
	"miento", "mientos", "menta", "mento",
	"idad", "idades", "ible", "ibles",
	"ismo", "ismos", "ista", "istas",
	"ivo", "iva", "ivos", "ivas",
	"aje", "ajes", "eza", "ezas",
	"aba", "ada", "ido",
	"ar", "er", "ir", "as", "es",
	"an", "en", "in", "os",
	"a", "o", "e", "s",
}

func stemSpanish(word string) string {
	// Don't stem short words
	if len(word) <= 3 {
		return word
	}
	lower := strings.ToLower(word)
	// Try to strip suffixes (longest first)
	for _, suffix := range spanishSuffixes {
		if strings.HasSuffix(lower, suffix) {
			stem := lower[:len(lower)-len(suffix)]
			if len(stem) >= 2 {
				return stem
			}
		}
	}
	return lower
}

func tokenize(text string) []string {
	var tokens []string
	var buf strings.Builder

	text = strings.ToLower(text)

	for i, r := range text {
		// Keep letters, digits, underscores, and hyphens inside words
		if r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) {
			buf.WriteRune(r)
		} else if r == '-' && i > 0 && i < len(text)-1 {
			// Hyphen: keep if it looks like part of a compound word
			prev := rune(text[i-1])
			next := rune(text[i+1])
			if (unicode.IsLetter(prev) || unicode.IsDigit(prev)) &&
				(unicode.IsLetter(next) || unicode.IsDigit(next)) {
				buf.WriteRune(r)
				continue
			}
			// Otherwise treat as separator
			if buf.Len() > 1 {
				tokens = append(tokens, buf.String())
			}
			buf.Reset()
		} else {
			if buf.Len() > 1 {
				tokens = append(tokens, buf.String())
			}
			buf.Reset()
		}
	}
	if buf.Len() > 1 {
		tokens = append(tokens, buf.String())
	}

	// Apply Spanish stemming to all tokens
	stemmed := make([]string, 0, len(tokens))
	for _, t := range tokens {
		s := stemSpanish(t)
		if s != "" {
			stemmed = append(stemmed, s)
		}
	}

	return stemmed
}

func countTokens(tokens []string) map[string]int {
	freqs := make(map[string]int, len(tokens))
	for _, t := range tokens {
		freqs[t]++
	}
	return freqs
}
