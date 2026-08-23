package main

import (
	"strings"
	"unicode"
)

// ── Normalización de acentos ──
// Mapea caracteres acentuados a su forma ASCII para que
// "configuración" == "configuracion" y "año" == "ano".
var accentMap = map[rune]rune{
	'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
	'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
	'ü': 'u', 'Ü': 'U', 'ñ': 'n', 'Ñ': 'N',
}

func normalize(s string) string {
	return strings.Map(func(r rune) rune {
		if n, ok := accentMap[r]; ok {
			return n
		}
		return r
	}, s)
}

// ── Stop words (español + inglés de uso seguro) ──
// Palabras gramaticales que solo inflan el índice y no aportan
// señal de búsqueda. Se excluyen términos relevantes para código
// (get, set, user, data, file, api, error, list, map, test, ...).
var spanishStopWords = map[string]bool{
	// artículos / determinantes
	"el": true, "la": true, "los": true, "las": true, "lo": true,
	"les": true, "le": true, "un": true, "una": true, "unos": true, "unas": true,
	// preposiciones
	"de": true, "del": true, "a": true, "en": true, "por": true, "para": true,
	"con": true, "sin": true, "al": true, "ante": true, "bajo": true,
	"contra": true, "desde": true, "durante": true, "hacia": true, "hasta": true,
	"mediante": true, "segun": true, "sobre": true, "tras": true, "via": true,
	// conjunciones
	"y": true, "e": true, "o": true, "u": true, "pero": true, "porque": true,
	"que": true, "pues": true, "si": true, "aunque": true, "como": true,
	"cuando": true, "donde": true, "mientras": true, "salvo": true, "excepto": true,
	// pronombres
	"yo": true, "tu": true, "tú": true, "ella": true, "ello": true,
	"nos": true, "nosotros": true, "vos": true, "vosotros": true, "ustedes": true,
	"me": true, "te": true, "se": true, "os": true, "mi": true,
	"mis": true, "tus": true, "su": true, "sus": true, "este": true,
	"esta": true, "esto": true, "ese": true, "esa": true, "eso": true, "aquel": true,
	"aquella": true, "aquello": true, "quien": true, "cual": true, "cuyo": true,
	// ser / estar / haber (formas frecuentes)
	"es": true, "son": true, "fue": true, "fueron": true, "era": true, "eran": true,
	"soy": true, "eres": true, "somos": true, "estan": true,
	"estaba": true, "hay": true, "ha": true, "han": true, "he": true, "hemos": true,
	"haber": true, "ser": true, "estar": true, "sido": true, "siendo": true,
	// adverbios / varios
	"muy": true, "mas": true, "menos": true, "todo": true, "toda": true, "todos": true,
	"todas": true, "nada": true, "algo": true, "cada": true, "otro": true, "otra": true,
	"otros": true, "otras": true, "mismo": true, "misma": true, "entre": true,
	"asi": true, "tambien": true, "tampoco": true, "aqui": true, "alli": true,
	"alla": true, "ya": true, "aun": true, "todavia": true, "solo": true, "sola": true,
	"no": true, "puede": true, "debe": true, "podria": true,
	// inglés común (seguro en código)
	"the": true, "and": true, "for": true, "are": true, "you": true, "our": true,
	"out": true, "not": true, "but": true, "with": true, "this": true, "that": true,
}

// ── Spanish stemmer ──
// Lightweight suffix-stripping stemmer. Not a full Snowball stemmer,
// but significantly reduces inflectional variance for Spanish text.
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

// splitIdentifier divides a raw word into sub-tokens on:
//   - camelCase boundaries  ("getUserData" -> ["get", "User", "Data"])
//   - letter<->digit boundaries ("v2api" -> ["v", "2", "api"])
//   - "HTTP" -> "Server" splits before the last uppercase of an acronym run
// Underscores and hyphens are already removed by the caller (tokenize),
// so this only needs to handle in-word case/digit transitions.
func splitIdentifier(word string) []string {
	runes := []rune(word)
	if len(runes) == 0 {
		return nil
	}
	var parts []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			parts = append(parts, cur.String())
			cur.Reset()
		}
	}
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		if i > 0 {
			prev := runes[i-1]
			switch {
			case unicode.IsLower(prev) && unicode.IsUpper(r):
				flush() // get|User
			case unicode.IsLetter(prev) && unicode.IsDigit(r):
				flush() // v|2
			case unicode.IsDigit(prev) && unicode.IsLetter(r):
				flush() // 2|api
			case unicode.IsUpper(prev) && unicode.IsUpper(r) && i+1 < len(runes) && unicode.IsLower(runes[i+1]):
				flush() // HTTP|Server
			}
		}
		cur.WriteRune(r)
	}
	flush()
	return parts
}

func tokenize(text string) []string {
	// 1) Normalize accents (preserve case so camelCase is detectable).
	text = normalize(text)

	// 2) Extract raw words (letters + digits only; _ and - are separators).
	var rawWords []string
	var buf strings.Builder
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			buf.WriteRune(r)
		} else if buf.Len() > 0 {
			rawWords = append(rawWords, buf.String())
			buf.Reset()
		}
	}
	if buf.Len() > 0 {
		rawWords = append(rawWords, buf.String())
	}

	// 3) Split camelCase / digit boundaries, lowercase, drop stop words, stem.
	out := make([]string, 0, len(rawWords))
	for _, raw := range rawWords {
		for _, part := range splitIdentifier(raw) {
			low := strings.ToLower(part)
			if len(low) < 2 {
				continue
			}
			if spanishStopWords[low] {
				continue
			}
			stemmed := stemSpanish(low)
			if stemmed != "" && !spanishStopWords[stemmed] {
				out = append(out, stemmed)
			}
		}
	}
	return out
}

func countTokens(tokens []string) map[string]int {
	freqs := make(map[string]int, len(tokens))
	for _, t := range tokens {
		freqs[t]++
	}
	return freqs
}
