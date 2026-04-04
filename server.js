'use strict';

const express = require('express');
const path    = require('path');
const { GoogleGenAI }   = require('@google/genai');
const { algoliasearch } = require('algoliasearch');
require('dotenv').config();

// ─── Algolia client ───────────────────────────────────────────────────────────
const algolia = algoliasearch(
  process.env.ALGOLIA_APP_ID,
  process.env.ALGOLIA_API_KEY
);

const app = express();
const ai  = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// ─── Gemini prompt: strip filler, return clean query ─────────────────────────
const buildPrompt = (raw) =>
`You are an e-commerce search query cleaner.
Convert the raw voice transcript into a concise product search query.
Strip ALL filler words, greetings, and conversational phrases.
Return ONLY the cleaned query — no explanation, no quotes, no trailing punctuation.

Examples:
"um show me wireless noise cancelling headphones please" → wireless noise-cancelling headphones
"I'm looking for red running shoes maybe size 10"        → red running shoes size 10
"can you get me some fresh organic apples"               → organic apples
"hey find me a black leather wallet under fifty dollars" → black leather wallet under $50
"I want to buy like a good gaming mouse"                 → gaming mouse

Raw: ${raw}
Cleaned:`;

// ─── Middleware ───────────────────────────────────────────────────────────────
app.use(express.static(path.join(__dirname)));
app.use(express.json());

// ─── Clean endpoint ───────────────────────────────────────────────────────────
app.post('/api/clean', async (req, res) => {
  const { query } = req.body ?? {};
  if (!query?.trim()) return res.json({ cleaned: '' });

  try {
    const result  = await ai.models.generateContent({
      model    : 'gemini-2.5-flash',
      contents : buildPrompt(query.trim()),
    });

    // result.text is a getter on the GenerateContentResponse object
    const cleaned = (result.text ?? '').trim() || query.trim();
    console.log(`[Gemini]  "${query}"  →  "${cleaned}"`);
    res.json({ cleaned });

  } catch (err) {
    console.error('[Gemini] Error:', err.message);
    // Fallback: return raw query so search still works
    res.json({ cleaned: query.trim() });
  }
});

// ─── Algolia search endpoint ──────────────────────────────────────────────────
app.get('/api/search', async (req, res) => {
  const q = (req.query.q || '').trim();
  if (!q) return res.json({ hits: [], nbHits: 0 });

  try {
    const result = await algolia.searchSingleIndex({
      indexName    : process.env.ALGOLIA_INDEX_NAME,
      searchParams : {
        query               : q,
        hitsPerPage         : 9,
        attributesToRetrieve: [
          'objectID', 'name', 'brand_name', 'category_name',
          'subcategory_name', 'item_price', 'images_full_url',
          'avg_rating', 'rating_count', 'slug',
        ],
      },
    });

    console.log(`[Algolia]  "${q}"  →  ${result.nbHits} hits`);
    res.json({ hits: result.hits, nbHits: result.nbHits });

  } catch (err) {
    console.error('[Algolia] Error:', err.message);
    res.status(500).json({ error: err.message, hits: [], nbHits: 0 });
  }
});

app.get('/health', (_req, res) => res.json({ ok: true }));

// ─── Start ────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;

if (process.env.NODE_ENV !== 'production' || !process.env.VERCEL) {
  app.listen(PORT, () => {
    const ok = (v) => v ? '\x1b[32m✓ Set\x1b[0m' : '\x1b[31m✗ Missing\x1b[0m';
    console.log(`
  ┌──────────────────────────────────────────────┐
  │   🎤  Voice Search                           │
  ├──────────────────────────────────────────────┤
  │   http://localhost:${PORT}                      │
  │   Gemini   : ${ok(process.env.GEMINI_API_KEY)}                     │
  │   Algolia  : ${ok(process.env.ALGOLIA_APP_ID)}                     │
  │   Index    : ${process.env.ALGOLIA_INDEX_NAME || '(unset)'}              │
  └──────────────────────────────────────────────┘
    `);
  });
}

module.exports = app;

