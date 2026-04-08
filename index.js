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

// ─────────────────────────────────────────────────────────────────────────────
//  GROQ PROMPT  (llama-3.1-8b-instant)
//  Kept under 100 tokens — TTFT on llama scales with input length.
//  Every extra token here costs real ms. No fluff, just the signal.
// ─────────────────────────────────────────────────────────────────────────────
const GROQ_SYSTEM =
`Baby product voice search cleaner. Output English search term only — no explanation.
Strip filler + particles: um uh hey please yaar bhai mujhe chahiye ka ki ke ko se mein par bhi wala wali
Translate: doodh→milk | langot→cloth nappy | chusni→pacifier | daant→teething | tel→massage oil | bada→large | ek→1 | do→2 | teen→3
Keep brand names and sizes exactly as spoken.
doodh ka bottle → milk feeding bottle
baby ki langot do pack → cloth nappy 2 pack
Pampers size 2 chahiye → Pampers size 2`;

const buildGroqMessages = (raw) => [
  { role: 'system', content: GROQ_SYSTEM },
  { role: 'user',   content: `Raw: ${raw}\nCleaned:` },
];

// ─────────────────────────────────────────────────────────────────────────────
//  GEMINI PROMPT  (gemini-2.5-flash)
//  Contextual + nuanced. Gemini understands intent — keep the smart rules
//  (symptom mapping, regional inference) but trim token count.
// ─────────────────────────────────────────────────────────────────────────────
const GEMINI_SYSTEM =
`You are a multilingual voice search query cleaner for an Indian baby care & maternity quick-commerce app (products for pregnant moms and babies 0–5 years).
Users speak English, Hindi, or Hinglish. Convert the raw voice transcript into the shortest precise English product search query for a baby catalogue.

Rules:
1. Strip ALL filler and Hindi grammatical particles (um, uh, yaar, bhai, mujhe, chahiye, ka, ki, ke, ko, se, wala/wali, etc.)
2. Translate every Hindi/regional baby or maternity term to its English catalogue equivalent.
3. If the user describes a symptom or need, map it to the most relevant product (e.g. "baby ko neend nahi aati" → baby sleep aid gripe water).
4. Preserve brand names, size, age range, stage, and variants exactly.
5. Do NOT add words the user did not say (except symptom-to-product mapping).
6. Return ONLY the cleaned English query — no explanation, no quotes.

Key translations: doodh→milk, langot→cloth nappy, chusni→pacifier, daant→teething,
tel maalish→baby massage oil, angochha→baby towel, jhula→baby swing,
delivery ke baad→postnatal, stan→nursing pad, pump→breast pump.

Examples:
"doodh ka bottle chahiye"              → milk feeding bottle
"mujhe Pampers size 2 chahiye"         → Pampers diaper size 2
"baby ki langot do pack"               → cloth nappy 2 pack
"daant nikal rahe hain kuch dena"      → teething gel
"baby ko raat ko neend nahi aati"      → baby sleep aid gripe water
"Huggies sensitive wipes ek bada pack" → Huggies sensitive baby wipes large pack
"prenatal vitamins folic acid wali"    → prenatal vitamins folic acid
"stage 1 formula milk Nan Pro"         → Nan Pro stage 1 infant formula`;

const buildGeminiPrompt = (raw) => `${GEMINI_SYSTEM}\n\nRaw: ${raw}\nCleaned:`;

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

// ─── Shared Algolia search helper ────────────────────────────────────────────
const ALGOLIA_ATTRS = [
  'objectID', 'name', 'brand_name', 'category_name',
  'subcategory_name', 'item_price', 'images_full_url',
  'avg_rating', 'rating_count', 'slug',
];

async function algoliaSearch(q, page = 0) {
  const result = await algolia.searchSingleIndex({
    indexName    : process.env.ALGOLIA_INDEX_NAME,
    searchParams : { query: q, hitsPerPage: 10, page, attributesToRetrieve: ALGOLIA_ATTRS },
  });
  return { hits: result.hits, nbHits: result.nbHits };
}

const HINGLISH_STOP_WORDS = new Set([
  // Hinglish
  'yaar', 'bhai', 'mujhe', 'chahiye', 'laao', 'dikhao', 'wala', 'wali', 'ka', 'ki', 'ke', 'ko', 'se', 'mein', 'par', 'kya', 'hai', 'koi', 'kuch', 'aur', 'usko',
  // English
  'for', 'to', 'with', 'the', 'a', 'an', 'in', 'on', 'at', 'of', 'and', 'my', 'is', 'i', 'want', 'need', 'show', 'get', 'give'
]);

function shouldRefine(raw) {
  const words = raw.split(/\s+/);
  if (words.length >= 4) return true; // Anything long gets LLM

  for (const w of words) {
    if (HINGLISH_STOP_WORDS.has(w.toLowerCase())) return true;
  }
  return false;
}

// ─── Streaming Voice Search Endpoint ──────────────────────────────────────────
// One connection handles immediate raw results and delayed refined results
app.get('/api/voice-stream', async (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform', // Bypass Vercel/Express buffering
    'Connection': 'keep-alive'
  });

  const raw = (req.query.q || '').trim();
  if (!raw) {
    res.write(`data: ${JSON.stringify({ step: 'error', error: 'Empty query' })}\n\n`);
    return res.end();
  }

  // 1. Backend evaluates query
  const needsRefinement = shouldRefine(raw);

  // 2. Start initial raw search
  try {
    const { hits, nbHits } = await algoliaSearch(raw, 0);
    res.write(`data: ${JSON.stringify({ step: 'raw', hits, nbHits, isRefining: needsRefinement })}\n\n`);
  } catch (err) {
    console.error('[Algolia Raw] Error:', err.message);
    res.write(`data: ${JSON.stringify({ step: 'raw', hits: [], nbHits: 0, isRefining: needsRefinement })}\n\n`);
  }

  // 3. Keep connection open and do LLM refinement if needed
  if (needsRefinement) {
    try {
      // Check cache first
      let cleaned = cleanCache.get(raw.toLowerCase());
      if (!cleaned) {
        cleaned = await cleanQuery(raw); // Calls Groq
        cacheSet(raw.toLowerCase(), cleaned);
      } else {
        console.log(`[Cache]   "${raw}"  →  "${cleaned}"`);
      }

      // If the LLM realized it didn't need to change anything:
      if (cleaned.toLowerCase() === raw.toLowerCase()) {
         res.write(`data: ${JSON.stringify({ step: 'refined', hits: null, cleaned })}\n\n`);
      } else {
         const { hits, nbHits } = await algoliaSearch(cleaned, 0);
         console.log(`[Algolia] Refined "${cleaned}" → ${nbHits} hits`);
         res.write(`data: ${JSON.stringify({ step: 'refined', hits, nbHits, cleaned })}\n\n`);
      }
    } catch (err) {
      console.error('[Refine] Error:', err.message);
      // Failsafe: tell UI refinement failed, stop spinning
      res.write(`data: ${JSON.stringify({ step: 'refined', hits: null, error: true })}\n\n`);
    }
  }

  // End the connection gracefully
  res.end();
});

// ─── Clean + search endpoint (Legacy, kept for backup) ────────────────────────
// Returns { cleaned, hits, nbHits } so the client never needs a second fetch.
app.post('/api/clean', async (req, res) => {
  const { query } = req.body ?? {};
  if (!query?.trim()) return res.json({ cleaned: '', hits: [], nbHits: 0 });

  const raw = query.trim().toLowerCase();
  let cleaned = cleanCache.get(raw);
  
  if (cleaned) {
    console.log(`[Cache]   "${raw}"  →  "${cleaned}"`);
  } else {
    try {
      cleaned = await cleanQuery(query.trim());
      cacheSet(raw, cleaned);
    } catch (err) {
      console.error('[Clean] All providers failed:', err.message);
      cleaned = query.trim();
    }
  }

  try {
    const { hits, nbHits } = await algoliaSearch(cleaned, 0);
    res.json({ cleaned, hits, nbHits });
  } catch (err) {
    console.error('[Algolia] Error:', err.message);
    res.json({ cleaned, hits: [], nbHits: 0 });
  }
});

// ─── Algolia search endpoint (used for the fast raw/quick query & pagination) ─
app.get('/api/search', async (req, res) => {
  const q = (req.query.q || '').trim();
  const page = parseInt(req.query.page) || 0;
  if (!q) return res.json({ hits: [], nbHits: 0 });

  try {
    const { hits, nbHits } = await algoliaSearch(q, page);
    console.log(`[Algolia]  "${q}" (page ${page}) →  ${nbHits} hits`);
    res.json({ hits, nbHits });
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

