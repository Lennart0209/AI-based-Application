# BMW Sustainability Report Extractor

**Projektseminar: AI Seminar – Sommersemester 2025**  
**Leuphana Universität Lüneburg**  
**Betreuung:** Dr. Debayan Banerjee, Prof. Dr. Ricardo Usbeck

---

## Projektziel

Ziel dieses Projekts ist es, einen interaktiven PDF-Analysator zu entwickeln, der mithilfe eines Large Language Models (LLM) Nachhaltigkeitsberichte automatisch verarbeitet und relevante Umweltinformationen extrahiert. Die extrahierten Daten werden im strukturierten JSON-Format bereitgestellt.

Das Beispielprojekt basiert auf dem **BMW Group Sustainability Report 2022**.

---

## Funktionen

- Upload von PDF-Dateien (Nachhaltigkeitsbericht)
- Extraktion des Textes aus dem PDF
- Analyse der Inhalte durch GPT-4 (OpenAI API oder GWDG)
- Ausgabe eines strukturierten JSON mit den folgenden Feldern:

```json
{
  "name": "...",
  "CO2": "...",
  "NOX": "...",
  "Number_of_Electric_Vehicles": "...",
  "Impact": "...",
  "Risks": "...",
  "Opportunities": "...",
  "Strategy": "...",
  "Actions": "...",
  "Adopted_policies": "...",
  "Targets": "..."
}
