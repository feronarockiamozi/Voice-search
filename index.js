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
//  Mechanical + rigid. Llama needs explicit rules and zero ambiguity.
//  Uses system/user roles — Llama follows system instructions more strictly.
// ─────────────────────────────────────────────────────────────────────────────
const GROQ_SYSTEM = `You convert Hinglish/Hindi baby product voice queries into short English search terms.

OUTPUT RULES — follow every one, no exceptions:
1. Output MUST be English only. Zero Hindi/Urdu words allowed in the output.
2. Remove every Hindi grammatical particle: ka ki ke ko se mein par pe ne toh bhi hi na aur phir woh yeh jo
3. Remove ALL filler: um uh er hey please yaar bhai arre mujhe humein chahiye dena dikhao lao
4. Translate every Hindi product word using the table below. If a Hindi word is not in the table, remove it.
5. Keep brand names exactly (Pampers, Huggies, Himalaya, Johnson's, Pigeon, Chicco, Mamy Poko, Nan Pro).
6. Keep size/quantity/age in English (size 2, newborn, 0-6 months, 200ml, pack of 50, large).
7. Return ONLY the final search term — no explanation, no punctuation at the end.

TRANSLATION TABLE:
doodh / dudh    → milk
langot          → cloth nappy
chusni          → pacifier
jhula / palna   → baby swing
tel / maalish   → massage oil
angochha        → baby towel
daant           → teething
bottle          → feeding bottle
katori chamach  → weaning bowl spoon
stan            → nursing pad
pump            → breast pump
bada            → large
chota           → small
naya            → new
ek              → 1
do              → 2
teen            → 3
char            → 4
paanch          → 5

EXAMPLES (study the particle removal carefully):
doodh ka bottle            → milk feeding bottle
baby ki langot do pack     → cloth nappy 2 pack
chusni newborn ke liye     → newborn pacifier
Pampers size 2 chahiye     → Pampers size 2
Himalaya tel 200ml wala    → Himalaya massage oil 200ml
Huggies wipes ek bada pack → Huggies wipes large pack
daant nikal rahe hain      → teething gel
Nan Pro stage 1 doodh      → Nan Pro stage 1 milk formula`;

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

