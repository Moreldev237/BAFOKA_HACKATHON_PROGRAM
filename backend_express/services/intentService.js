export async function detectIntent(text, lang = "fr") {

  const prompts = {
    fr: `
Tu es un moteur d'intentions très strict.
Analyse le texte utilisateur et retourne UNE intention parmi :

- inscription
- connexion
- ouvrir
- paiement
- information
- résiliation
- retour
- moins_35
- entre_35_50
- plus_50
- homme
- femme
- autre

Réponds uniquement par un mot.


Exemples : 
"Je_suis_jeune" -> moins_35
"Je_suis_adulte" -> entre_35_50
"Je_suis_senior" -> plus_50
`,
    en: `
You are a very strict intent classifier.
Analyze the user text and return ONE of the following intents:

- signup
- enroll
- login
- payment
- info
- back
- under_35
- between_35_50
- over_50
- man
- woman
- previously
- cancel
- other

Respond with only one word.


Examples: 
"I_am_young" -> under_35
"I_am_adult" -> between_35_50
"I_am_senior" -> over_50
`
  };

  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`,
     // "HTTP-Referer": "http://localhost",
      "X-Title": "AudioAssistant",
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "openai/gpt-4o-mini",
      messages: [
        { role: "system", content: prompts[lang] || prompts.fr },
        { role: "user", content: text }
      ]
    })
  });

  const data = await response.json();
  return data.choices[0].message.content.trim().toLowerCase();
}
