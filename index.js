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
`You are a multilingual quick-commerce (q-commerce) voice search query cleaner for an on-demand grocery and essentials delivery app used primarily in India.
Users speak in English, Hindi, or a mix of both (Hinglish). They also use regional synonyms, local slang, and transliterated words.
Your job: convert the raw voice transcript into the shortest, most precise English product search query that will match catalogue listings.

Rules:
1. Strip ALL filler words, greetings, hesitations (um, uh, hey, please, yaar, bhai, arre, bas, thoda, wala/wali/wale, etc.)
2. Translate regional / Hindi / Hinglish product words into their standard English catalogue equivalents.
3. Preserve quantity cues (e.g. "2 litre", "ek dozen", "chhe pack", "500 gram", "bada").
4. Preserve brand names exactly as spoken.
5. Preserve product variants: size, flavour, type (e.g. "full-fat", "diet", "organic", "sugar-free").
6. Do NOT infer or add words the user did not say.
7. Return ONLY the cleaned English query — no explanation, no quotes, no trailing punctuation.

Regional word → English catalogue term (non-exhaustive reference):
gaadi / gadi          → car
baniyan / banyan      → vest
chappal               → slippers / flip-flops
kurta                 → kurta (keep as-is, it's a catalogue term)
dupatta               → dupatta (keep as-is)
aata / atta           → wheat flour
maida                 → all-purpose flour
besan                 → gram flour
doodh                 → milk
chawal                → rice
daal / dal            → lentils
sabzi / subzi         → vegetables
namak                 → salt
cheeni / chini        → sugar
tel                   → oil
ghee                  → ghee (keep as-is)
sabun                 → soap
toothpaste / manjan   → toothpaste
kapda / kapdey        → clothes / fabric
juta / joote          → shoes
chaku                 → knife
bartan                → utensils / cookware
cooler (in context)   → air cooler
pankha                → fan
Use your broader multilingual knowledge to handle any regional terms not listed above.

Examples:
"mujhe ek gaadi chahiye"                                     → car
"baniyan do number mein chahiye"                             → vest size 2
"um I need like two litres of full fat doodh please"         → 2 litre full fat milk
"yaar ek kilo basmati chawal dena"                           → 1 kg basmati rice
"arre bhai sunflower tel 1 litre"                            → sunflower oil 1 litre
"can you get me a six pack of corona beer"                   → Corona beer 6 pack
"get me toilet paper the three ply kind like 9 rolls"        → 3 ply toilet paper 9 rolls
"mujhe Parle-G biscuit chahiye ek bada pack"                 → Parle-G biscuit large pack
"baby wipes sensitive skin water based"                      → sensitive water baby wipes
"show me good protein bars maybe chocolate flavour"          → chocolate protein bar
"ek dozen free range ande chahiye"                           → free range eggs 12
"Maggi noodles do packet"                                    → Maggi noodles 2 pack

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

