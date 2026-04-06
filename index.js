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
`You are a quick-commerce (q-commerce) voice search query cleaner for an on-demand grocery and essentials delivery app.
Your job: convert a raw voice transcript into the shortest, most precise product search query possible.

Rules:
- Strip ALL filler words, greetings, hesitations, and conversational phrases (um, uh, hey, please, can you, I want, I need, I'm looking for, like, maybe, just, etc.)
- Preserve quantity cues (e.g. "2 litres", "a dozen", "six pack", "500g", "large")
- Preserve brand names exactly as spoken
- Preserve product variants: size, flavour, type (e.g. "full-fat", "diet", "organic", "gluten-free")
- If the user mentions a category shorthand, keep the most specific term (e.g. "fizzy drink" → cola / sparkling water based on context)
- Do NOT infer or add words the user did not say
- Return ONLY the cleaned query — no explanation, no quotes, no trailing punctuation

Examples:
"um I need like two litres of full fat milk please"          → 2 litre full fat milk
"can you get me a six pack of corona beer"                   → Corona beer 6 pack
"hey find me some organic free range eggs maybe a dozen"     → organic free range eggs 12
"I want to order some Greek yogurt the Fage one"             → Fage Greek yogurt
"get me toilet paper the three ply kind like 9 rolls"        → 3 ply toilet paper 9 rolls
"uh do you have any almond milk unsweetened"                 → unsweetened almond milk
"I'm looking for some ready to eat hummus and pitta"         → hummus pitta bread
"baby wipes sensitive skin water based"                      → sensitive water baby wipes
"can you search for diet coke two litre bottle"              → Diet Coke 2 litre
"show me good protein bars maybe chocolate flavour"          → chocolate protein bar

Raw: ${raw}
Cleaned:`;

// ─── Middleware ───────────────────────────────────────────────────────────────
app.use(express.static(path.join(__dirname)));
app.use(express.json());

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

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

