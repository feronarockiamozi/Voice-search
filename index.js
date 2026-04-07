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

// ─── Shared system prompt (used by both Groq and Gemini) ─────────────────────
const SYSTEM_PROMPT =
`You are a multilingual voice search query cleaner for an Indian quick-commerce app that sells ONLY baby care and maternity products — covering pregnant mothers and babies from newborn up to 5 years old.
Users speak in English, Hindi, or Hinglish and often use regional/informal terms for baby and pregnancy products.
Your job: convert the raw voice transcript into the shortest, most precise English product search query that will match catalogue listings on a baby care platform.

Rules:
1. Strip ALL filler words, greetings, hesitations (um, uh, yaar, bhai, arre, bas, thoda, wala/wali/wale, please, can you, I need, mujhe chahiye, etc.)
2. Translate Hindi / Hinglish / regional baby and maternity terms into their standard English catalogue equivalents (see reference below).
3. Preserve quantity and size cues (e.g. "size 2", "newborn", "0-6 months", "100ml", "pack of 50").
4. Preserve brand names exactly as spoken (e.g. Pampers, Huggies, Himalaya, Mamy Poko, Johnson's, Chicco, Pigeon).
5. Preserve product variants: age range, stage, type, flavour (e.g. "stage 1", "sensitive skin", "fragrance-free", "apple flavour").
6. If the user describes a symptom or need (e.g. "baby ko neend nahi aati"), map it to the most relevant product category (e.g. → baby sleep aid / gripe water).
7. Do NOT infer or add words the user did not say beyond the symptom-to-product mapping above.
8. Return ONLY the cleaned English query — no explanation, no quotes, no trailing punctuation.

Regional / Hindi word → English catalogue term (baby & maternity focus):
langot → cloth nappy | jhula/palna → baby swing | chusni → pacifier
doodh/dudh → milk | daant nikalna → teething | tel maalish → baby massage oil
angochha → baby towel | bottle (baby) → feeding bottle | katori-chamach → weaning set
delivery ke baad → postnatal | stan/breast → nursing pad | pump → breast pump
Use your broader multilingual knowledge to handle any regional terms not listed above.

Examples:
"mujhe Pampers size 2 chahiye" → Pampers diaper size 2
"baby ka langot do pack" → cloth nappy 2 pack
"um daant nikal rahe hain kuch dena" → teething gel
"baby ko raat ko neend nahi aati" → baby sleep aid gripe water
"Huggies sensitive wipes ek bada pack" → Huggies sensitive baby wipes large pack
"stage 1 formula milk Nan Pro" → Nan Pro stage 1 infant formula`;

// Groq uses chat messages; Gemini takes a single string
const buildGroqMessages = (raw) => [
  { role: 'user', content: `${SYSTEM_PROMPT}\n\nRaw: ${raw}\nCleaned:` },
];

const buildGeminiPrompt = (raw) => `${SYSTEM_PROMPT}\n\nRaw: ${raw}\nCleaned:`;

// ─── Groq call (primary — fast, ~100-200ms) ───────────────────────────────────
async function callGroq(raw) {
  const controller = new AbortController();
  const timeout    = setTimeout(() => controller.abort(), 4000); // 4s timeout

  try {
    const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method  : 'POST',
      headers : {
        'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
        'Content-Type' : 'application/json',
      },
      body   : JSON.stringify({
        model       : 'llama-3.1-8b-instant',
        messages    : buildGroqMessages(raw),
        temperature : 0,
        max_tokens  : 60,
      }),
      signal : controller.signal,
    });

    if (!res.ok) throw new Error(`Groq HTTP ${res.status}`);
    const data = await res.json();
    return data.choices?.[0]?.message?.content?.trim() || raw;

  } finally {
    clearTimeout(timeout);
  }
}

// ─── Gemini call (fallback) ───────────────────────────────────────────────────
async function callGemini(raw) {
  const result = await ai.models.generateContent({
    model    : 'gemini-2.5-flash',
    contents : buildGeminiPrompt(raw),
  });
  return (result.text ?? '').trim() || raw;
}

// ─── Clean with Groq → Gemini fallback ───────────────────────────────────────
async function cleanQuery(raw) {
  if (process.env.GROQ_API_KEY) {
    try {
      const cleaned = await callGroq(raw);
      console.log(`[Groq]    "${raw}"  →  "${cleaned}"`);
      return cleaned;
    } catch (err) {
      console.warn(`[Groq] Failed (${err.message}), falling back to Gemini`);
    }
  }
  const cleaned = await callGemini(raw);
  console.log(`[Gemini]  "${raw}"  →  "${cleaned}"`);
  return cleaned;
}

// ─── Middleware ───────────────────────────────────────────────────────────────
app.use(express.static(path.join(__dirname)));
app.use(express.json());

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// ─── Gemini response cache (in-memory, max 500 entries) ──────────────────────
const cleanCache = new Map();
const CACHE_MAX  = 500;

function cacheSet(key, value) {
  if (cleanCache.size >= CACHE_MAX) {
    // evict oldest entry
    cleanCache.delete(cleanCache.keys().next().value);
  }
  cleanCache.set(key, value);
}

// ─── Clean endpoint ───────────────────────────────────────────────────────────
app.post('/api/clean', async (req, res) => {
  const { query } = req.body ?? {};
  if (!query?.trim()) return res.json({ cleaned: '' });

  const raw = query.trim().toLowerCase();

  // Cache hit — skip LLM entirely
  if (cleanCache.has(raw)) {
    console.log(`[Cache]   "${raw}"  →  "${cleanCache.get(raw)}"`);
    return res.json({ cleaned: cleanCache.get(raw) });
  }

  try {
    const cleaned = await cleanQuery(query.trim());
    cacheSet(raw, cleaned);
    res.json({ cleaned });
  } catch (err) {
    console.error('[Clean] All providers failed:', err.message);
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

