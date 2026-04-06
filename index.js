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
langot                      → cloth nappy / cloth diaper
nappy / napkin (baby)       → diaper
looper / lapet              → swaddle wrap
jhula / palna               → baby swing / baby cradle
tel maalish / malish        → baby massage oil
kajal (baby)                → baby kohl / baby kajal
doodh / dudh                → milk / infant formula
daant nikalna               → teething
colic / gas (baby context)  → gripe water / colic drops
angochha / angocha          → baby towel / hooded towel
rassi / naal                → umbilical cord care
susu / potty (baby context) → diaper rash cream / potty training
bottle (baby context)       → feeding bottle
nipple / teat               → bottle nipple / feeding teat
chusni / pacifier           → pacifier / soother
katori-chamach              → weaning bowl and spoon set
pregnancy belt / petti      → maternity support belt
delivery ke baad            → postpartum / postnatal
garbhavati / pregnant       → maternity / prenatal
stan / breast (nursing)     → nursing pad / breast pad
pump (nursing context)      → breast pump
Use your broader multilingual and medical knowledge to handle any regional terms not listed above.

Examples:
"mujhe Pampers size 2 chahiye"                                     → Pampers diaper size 2
"baby ka langot do pack"                                           → cloth nappy 2 pack
"um daant nikal rahe hain kuch dena"                               → teething gel / teething toy
"Himalaya baby tel 200ml"                                          → Himalaya baby massage oil 200ml
"chusni chahiye newborn ke liye"                                   → newborn pacifier soother
"Johnson's baby shampoo no tears wala"                             → Johnson's baby shampoo no tears
"mujhe breast pump chahiye electric"                               → electric breast pump
"baby ko raat ko neend nahi aati kuch hai kya"                     → baby sleep aid gripe water
"Huggies sensitive wipes ek bada pack"                             → Huggies sensitive baby wipes large pack
"prenatal vitamins folic acid wali"                                → prenatal vitamins folic acid
"stage 1 formula milk Nan Pro"                                     → Nan Pro stage 1 infant formula
"feeding bottle anti colic Pigeon"                                 → Pigeon anti-colic feeding bottle
"colic drop Woodward's"                                            → Woodward's gripe water colic drops
"baby food 6 month apple puree"                                    → apple puree baby food 6 months
"maternity pad delivery ke baad"                                   → maternity pad postnatal

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

