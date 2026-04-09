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
//  Budget: Groq is hard-capped at 500ms. Input length has negligible effect on
//  TTFT for instant models (~750 tok/s), so we can afford comprehensive rules.
//  8B models rely heavily on few-shot examples — keep those diverse & precise.
// ─────────────────────────────────────────────────────────────────────────────
const GROQ_SYSTEM =
`Baby & maternity e-commerce search optimizer (Pakistan market). Input: raw Hinglish/English voice transcript. Output: clean English product keywords only — no explanation, no quotes, no punctuation.

STRIP all Hindi/Urdu particles and filler (never include in output):
ko ka ki ke se mein par hai hain ne mujhe humein chahiye laao dikhao dena wala wali yaar bhai arre um uh please "show me" "i want" "give me" need looking

TRANSLATE (Hindi/Urdu → English):
doodh→milk, langot→cloth nappy, chusni→pacifier, jhula→baby swing, tel→oil, maalish→massage, bada→large, chota→small, ek→1, do→2, teen→3, char→4, stan→breast, angochha→towel, topi→cap, jurab→socks, khilona→toy, gadda→mattress, takiya→pillow, razai→blanket, kapda→cloth, pani→water, sabun→soap, talc→powder

SYMPTOMS → PRODUCTS:
gas/colic/pet dard/pet phoolna → baby colic gripe water
teething/daant/daant nikalna → teething gel
rash/bottom laal/nappy rash → diaper rash cream
neend nahi/raat rota/so nahi raha → baby sleep aid
khansi/cough → baby cough syrup
bukhar/fever → baby fever paracetamol
ulti/vomiting → gripe water
cradle cap/sir pe chilka → cradle cap shampoo
stretch marks/nishan → stretch mark cream
delivery ke baad → postnatal
doodh nahi aa raha/breastfeed problem → lactation supplement
diaper bahut geela → overnight diaper night pants

BRAND CORRECTIONS (phonetic → correct):
mamma earth / mama earth → Mamaearth
seba med / sibamed → Sebamed
mamy poko / mami poko / mammy poko → MamyPoko Pants
hugges / huges / hugies → Huggies
pamper / pampars → Pampers
himalya / himlaya → Himalaya
mother spash / mother sparsh → Mother Sparsh
chicko / chico → Chicco
bio oil / biooil → Bio-Oil
prega news / pregnanews → Prega News
nan pro / nanpro → Nan Pro
lacto jen / lactogen → Lactogen

PRESERVE exactly: brand names, sizes (ml g kg oz), age ranges (0-6m 6-12m), stage numbers (stage 1 2 3), pack counts, variants (sensitive organic extra-care unscented)

EXAMPLES:
baby ko gas ho rahi hai → baby colic gripe water
mujhe Pampers size 2 chahiye → Pampers diaper size 2
doodh ka bottle 160ml → milk feeding bottle 160ml
Mamma Earth tea tree facewash → Mamaearth tea tree face wash
daant nikal rahe hain kuch do → teething gel
delivery ke baad belly tight karna → postpartum belly belt
Huggies sensitive wipes ek bada pack → Huggies sensitive baby wipes large pack
Nan Pro stage 1 formula milk → Nan Pro stage 1 infant formula
baby ko raat ko neend nahi aati → baby sleep aid
do pack langot → cloth nappy 2 pack
prenatal vitamins folic acid wali → prenatal vitamins folic acid
stretch marks wali cream for tummy → stretch mark cream
6 mahine ke baby ka khaana → stage 2 baby food 6 months
baby massage oil Himalaya 200ml → Himalaya baby massage oil 200ml
diaper bag backpack black color → diaper bag backpack black
breast pump electric chahiye → electric breast pump
baby ko bukhar hai kuch dena → baby fever paracetamol
Sebamed baby shampoo 150 ml → Sebamed baby shampoo 150ml
show me some baby wipes please → baby wipes
raat ko diaper leak ho raha hai → overnight diaper pants`;

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

// ─── Groq call (primary — fast, ~150-250ms) ──────────────────────────────────
async function callGroq(raw) {
  const controller = new AbortController();
  const timeout    = setTimeout(() => controller.abort(), 500); // hard cap: stay within latency budget

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

// ─── Clean with Groq → raw fallback (Gemini only when no Groq key) ───────────
// Groq is hard-capped at 500ms. On failure we return raw immediately rather
// than cascading to Gemini (which can take 1-3s and blows the latency budget).
async function cleanQuery(raw) {
  if (process.env.GROQ_API_KEY) {
    try {
      const cleaned = await callGroq(raw);
      console.log(`[Groq]    "${raw}"  →  "${cleaned}"`);
      return cleaned;
    } catch (err) {
      console.warn(`[Groq] Failed (${err.message}), using raw query`);
      return raw; // stay within latency budget
    }
  }
  // No Groq key configured — use Gemini
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

// ─── Golden Dictionary (Whitelist) ───────────────────────────────────────────
// If ALL words in a query are in this list, we skip LLM. 
// If ANY word is missing (like "kache"), we call LLM for refinement.
const GOLDEN_DICTIONARY = new Set([
  // PTypes
  'diaper', 'diapers', 'wipe', 'wipes', 'bottle', 'bottles', 'toy', 'toys', 
  'shoes', 'bag', 'bags', 'diaper bag', 'diaper bags', 'shampoo', 'soap', 'lotion',
  'cream', 'oil', 'powder', 'feeding', 'pacifier', 'nipple', 'formula',
  // Specific Samples
  'pampers', 'huggies', 'mamaearth', 'himalaya', 'sebamed', 'chicco', 'mee mee',
  'johnsons', 'aveeno', 'baby dove', 'mamy poko', 'mamypoko', 'libero'
]);

const HINGLISH_STOP_WORDS = new Set([
  'yaar', 'bhai', 'mujhe', 'chahiye', 'laao', 'dikhao', 'wala', 'wali', 'ka', 'ki', 'ke', 'ko', 'se', 'mein', 'par', 'kya', 'hai', 'koi', 'kuch', 'aur', 'usko',
  'for', 'to', 'with', 'the', 'a', 'an', 'in', 'on', 'at', 'of', 'and', 'my', 'is', 'i', 'want', 'need', 'show', 'get', 'give'
]);

function shouldRefine(raw) {
  const normalized = raw.toLowerCase().trim().replace(/[.,!?]+$/, '');
  const words = normalized.split(/\s+/);
  
  // Rule 1: Always refine conversational structure (4+ words)
  if (words.length >= 4) return true;

  // Rule 2: Trigger if ANY word matches a known stop word (conversational fluff)
  for (const w of words) {
    if (HINGLISH_STOP_WORDS.has(w)) return true;
  }

  // Rule 3: Whitelist check. 
  // If ANY word is NOT in the Golden Dictionary, we trigger the LLM.
  for (const w of words) {
    // Basic multi-word term check (e.g. "diaper bag")
    if (!GOLDEN_DICTIONARY.has(w)) {
      // Check if it's part of a multi-word entry in the dict
      const isPart = Array.from(GOLDEN_DICTIONARY).some(term => term.includes(w));
      if (!isPart) return true; // Unrecognized word -> REFINE!
    }
  }

  return false; // All words recognized as high-confidence English products
}

// ─── Voice Search Endpoint ────────────────────────────────────────────────────
// Decides if query needs LLM refinement, runs search once, returns final results.
app.get('/api/voice-stream', async (req, res) => {
  const raw = (req.query.q || '').trim();
  if (!raw) return res.json({ error: 'Empty query', hits: [], nbHits: 0 });

  let queryToSearch = raw;
  let wasRefined    = false;

  if (shouldRefine(raw)) {
    try {
      let cleaned = cleanCache.get(raw.toLowerCase());
      if (!cleaned) {
        cleaned = await cleanQuery(raw);
        cacheSet(raw.toLowerCase(), cleaned);
      } else {
        console.log(`[Cache]   "${raw}"  →  "${cleaned}"`);
      }
      if (cleaned.toLowerCase() !== raw.toLowerCase()) {
        queryToSearch = cleaned;
        wasRefined    = true;
      }
    } catch (err) {
      console.error('[Refine] Error:', err.message);
      // Fall through: search with raw query
    }
  }

  try {
    const { hits, nbHits } = await algoliaSearch(queryToSearch, 0);
    console.log(`[Algolia] "${queryToSearch}" → ${nbHits} hits`);
    res.json({ query: queryToSearch, hits, nbHits, wasRefined });
  } catch (err) {
    console.error('[Algolia] Error:', err.message);
    res.status(500).json({ error: err.message, hits: [], nbHits: 0 });
  }
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

