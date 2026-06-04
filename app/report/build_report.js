// Assemble le rapport .docx (Chapitres III & IV) a partir de results.json et des figures.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, LevelFormat, TableOfContents, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak, VerticalAlign,
} = require("docx");

const DIR = __dirname;
const FIG = path.join(DIR, "figures");
const R = JSON.parse(fs.readFileSync(path.join(DIR, "results.json"), "utf-8"));

const CW = 9026; // largeur utile A4, marges 1"
const NAVY = "1F4E79", LIGHT = "D5E8F0", GREY = "F2F4F6";

// ---------- helpers ----------
const fnum = (x, d = 0) => (x === null || x === undefined || isNaN(x)) ? "—" :
  Number(x).toLocaleString("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d });

function H(text, level) {
  const map = { 1: HeadingLevel.HEADING_1, 2: HeadingLevel.HEADING_2, 3: HeadingLevel.HEADING_3 };
  return new Paragraph({ heading: map[level], children: [new TextRun(text)] });
}
function P(text, opts = {}) {
  const runs = Array.isArray(text) ? text : [new TextRun({ text, ...opts })];
  return new Paragraph({ children: runs, spacing: { after: 120, line: 276 },
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED });
}
function bullet(text) {
  return new Paragraph({ numbering: { reference: "puces", level: 0 },
    children: [new TextRun(text)], spacing: { after: 60 } });
}
function caption(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text, italics: true, size: 18, color: "555555" })] });
}
function img(file, w) {
  const p = path.join(FIG, file);
  if (!fs.existsSync(p)) return P("[figure manquante: " + file + "]");
  // dimensions reelles via ratio connu approximatif: on borne la largeur
  const width = w || 600;
  // hauteur deduite par lecture rapide non triviale -> on fixe par convention selon le fichier
  const ratios = { "f5_roc.png": 0.8, "f6_shap.png": 0.49, "f7_km.png": 0.5,
                   "f8_cox.png": 0.54, "f2_heatmap.png": 0.40, "f3_abandon_wait.png": 0.43,
                   "f1_volume.png": 0.375, "f4_forecast.png": 0.40, "f9_staffing.png": 0.40 };
  const ratio = ratios[file] || 0.45;
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(p),
      transformation: { width, height: Math.round(width * ratio) },
      altText: { title: file, description: file, name: file } })] });
}
function table(headers, rows, widths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const mk = (txt, { head = false, w } = {}) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: { fill: head ? NAVY : "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 100, right: 100 }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: head ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text: String(txt), bold: head, color: head ? "FFFFFF" : "000000",
        size: 18 })] })],
  });
  const headRow = new TableRow({ tableHeader: true,
    children: headers.map((h, i) => mk(h, { head: true, w: widths[i] })) });
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      shading: { fill: ri % 2 ? GREY : "FFFFFF", type: ShadingType.CLEAR },
      margins: { top: 50, bottom: 50, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: String(c), size: 18 })] })],
    })),
  }));
  return new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths,
    rows: [headRow, ...bodyRows] });
}

// ---------- contenu dynamique ----------
const G = R.global;
const typeRows = R.kpi_type.map(t => [t.type_label, fnum(t.appels), fnum(t.taux_abandon, 2),
  fnum(t.aht_moyen, 0), fnum(t.attente_moyenne, 0)]);
const fcRows = R.forecast.map(f => [f.modele, fnum(f.MAE, 1), fnum(f.RMSE, 1), fnum(f["MAPE_%"], 2)]);
const abRows = R.abandon_table.map(a => [a.Modele, fnum(a.Accuracy, 3), fnum(a.Precision, 3),
  fnum(a.Recall, 3), fnum(a.F1, 3), fnum(a["AUC-ROC"], 3)]);
const coxRows = R.cox.filter(c => c.variable !== "Intercept").map(c =>
  [c.variable, fnum(c.HR, 3), `${fnum(c.HR_bas, 2)} – ${fnum(c.HR_haut, 2)}`, fnum(c.p_value, 4)]);
const kmRows = Object.entries(R.km_medians).map(([k, v]) => [k, fnum(v, 0)]);
const bestFc = R.forecast.slice().sort((a, b) => a["MAPE_%"] - b["MAPE_%"])[0];
const bestAb = R.abandon_table[0];

// ====================================================================
const children = [];

// --- Page de titre ---
children.push(
  new Paragraph({ spacing: { before: 1200, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Optimisation de la planification des effectifs", bold: true, size: 40, color: NAVY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: "par prévision du volume d'appels dans un centre de relation client — approche Business Intelligence et modèles prédictifs", size: 26, color: "333333" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "Chapitre III — Implémentation BI & Machine Learning", bold: true, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: "Chapitre IV — Résultats et recommandations", bold: true, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Corpus : AnonymousBank — centre d'appel bancaire, année 1999", italics: true, size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: `${fnum(G.appels_total)} appels analysés · ${R.abandon_infos ? "" : ""}taux d'abandon ${fnum(G.taux_abandon, 2)} %`, size: 20, color: "555555" })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// --- TOC ---
children.push(H("Table des matières", 1));
children.push(new TableOfContents("Sommaire", { hyperlink: true, headingStyleRange: "1-3" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ===================== CHAPITRE III =====================
children.push(H("Chapitre III — Implémentation BI et Machine Learning", 1));
children.push(P("Ce chapitre décrit la mise en œuvre opérationnelle du dispositif décisionnel, depuis l'ingestion des données transactionnelles brutes jusqu'aux modèles prédictifs et au moteur de dimensionnement des effectifs. L'implémentation suit la méthodologie CRISP-DM et s'appuie sur une architecture en couches (ETL, entrepôt, restitution analytique)."));

children.push(H("3.1 Chaîne ETL et qualité des données", 2));
children.push(P("Le corpus brut est un fichier transactionnel tabulé de 444 448 enregistrements couvrant l'année 1999. Chaque ligne correspond à un appel et comporte dix-sept variables : identifiants, priorité, type de service, horodatages des phases (serveur vocal interactif, file d'attente, service) et durées associées, issue de l'appel et agent affecté. La phase d'extraction-transformation-chargement (ETL) a comporté : (i) le rejet des lignes mal formées (délimiteurs intégrés, colonnes décalées) par filtrage sémantique ; (ii) la conversion des champs temporels au format H:MM:SS en secondes et la reconstruction d'un horodatage complet ; (iii) le recodage de l'issue d'appel ; (iv) le calcul de variables dérivées (heure, créneau de 30 minutes, jour de semaine, charge offerte)."));
children.push(P(`À l'issue du nettoyage, ${fnum(G.appels_total)} appels valides (hors ${fnum(G.appels_fantomes)} appels fantômes) ont été retenus, soit un taux de rétention proche de 100 %, attestant de la bonne qualité structurelle du corpus. Le taux d'abandon global s'établit à ${fnum(G.taux_abandon, 2)} %, le temps moyen de traitement (AHT) à ${fnum(G.aht_moyen_s, 0)} secondes, et le centre mobilise ${fnum(G.nb_agents)} agents.`));

children.push(H("3.2 Modèle multidimensionnel et data marts", 2));
children.push(P("Les données nettoyées alimentent un entrepôt structuré en schéma en étoile. La table de faits FAIT_APPELS (grain : un appel) porte les mesures quantitatives (durées VRU, attente, service ; indicateurs d'abandon) et se rattache à cinq dimensions : Temps (date, créneau, jour, mois), Type de service, Agent, Priorité et Issue. Une table de faits agrégée par créneau et type alimente spécifiquement les modules de prévision et de dimensionnement. Trois data marts métier en sont dérivés : Planification, Performance des agents, et Qualité & Abandon."));

children.push(H("3.3 Indicateurs de pilotage (tableau de bord exécutif)", 2));
children.push(P("Le tableau de bord exécutif synthétise les indicateurs clés de performance par type de service. Le tableau 3.1 met en évidence une forte hétérogénéité : les prospects (NW) présentent le taux d'abandon le plus élevé, tandis que les services à forte technicité (Internet) affichent les durées de traitement les plus longues."));
children.push(table(["Type de service", "Appels", "Abandon %", "AHT (s)", "Attente (s)"],
  typeRows, [3026, 1500, 1500, 1500, 1500]));
children.push(caption("Tableau 3.1 — Indicateurs de performance par type de service."));
children.push(img("f2_heatmap.png"));
children.push(caption("Figure 3.1 — Heatmap de la charge moyenne (jour de semaine × heure) : visualisation des pics et creux d'activité."));
children.push(img("f1_volume.png"));
children.push(caption("Figure 3.2 — Série quotidienne du volume d'appels (année 1999)."));

children.push(H("3.4 Module de prévision du volume", 2));
children.push(P("La prévision du volume quotidien repose sur la comparaison de plusieurs familles de modèles : un modèle de référence naïf saisonnier, un modèle statistique SARIMA capturant la saisonnalité hebdomadaire, le modèle additif Prophet, et un réseau de neurones récurrent LSTM (mémoire à long terme). L'évaluation est réalisée sur un horizon de test temporel (hold-out chronologique), garantissant l'absence de fuite d'information future. Les métriques retenues sont l'erreur absolue moyenne (MAE), la racine de l'erreur quadratique moyenne (RMSE) et l'erreur absolue moyenne en pourcentage (MAPE)."));

children.push(H("3.5 Module de prédiction de l'abandon", 2));
children.push(P("La prédiction de l'abandon est formulée comme un problème de classification binaire (appel abandonné vs servi) sur la population des appels mis en file d'attente. Afin d'éviter toute fuite d'information, le temps d'attente total — qui constitue, pour un appel abandonné, la patience effective du client — est exclu des variables explicatives. Les prédicteurs retenus sont disponibles à l'arrivée de l'appel : créneau horaire, jour, mois, week-end, type de service, priorité, charge offerte (congestion du créneau) et durée de navigation dans le serveur vocal. Quatre algorithmes sont comparés : régression logistique, forêt aléatoire, gradient boosting et XGBoost. Le déséquilibre des classes est traité par pondération."));
children.push(P("L'interprétabilité du meilleur modèle est assurée par les valeurs de Shapley (SHAP, Lundberg & Lee, 2017), qui décomposent chaque prédiction en contributions additives des variables, transformant le modèle « boîte noire » en recommandations actionnables."));

children.push(H("3.6 Analyse de survie et dimensionnement Erlang", 2));
children.push(P("Deux modules complètent le dispositif. D'une part, une analyse de survie de la patience client (Brown et al., 2005) modélise non pas seulement si, mais quand un client abandonne : la durée étudiée est le temps d'attente, l'événement est l'abandon, et les appels servis sont traités comme censurés à droite. La fonction de survie est estimée par Kaplan-Meier et les facteurs de risque par un modèle de Cox à hasards proportionnels. D'autre part, le moteur de dimensionnement traduit la charge prévue en nombre d'agents requis par créneau, via la théorie des files d'attente (formule d'Erlang C), corrigée par une approximation d'Erlang A intégrant l'impatience des clients."));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ===================== CHAPITRE IV =====================
children.push(H("Chapitre IV — Résultats et recommandations", 1));
children.push(P("Ce chapitre présente les résultats empiriques obtenus pour chacun des trois objectifs spécifiques, puis en dérive des recommandations managériales actionnables."));

children.push(H("4.1 Résultats de la prévision du volume", 2));
children.push(P(`Le tableau 4.1 compare les modèles sur la période de test. Le modèle ${bestFc.modele} obtient la meilleure exactitude, avec une MAPE de ${fnum(bestFc["MAPE_%"], 2)} %. De manière notable, le modèle statistique SARIMA surpasse le réseau de neurones LSTM. Ce résultat, loin d'être contre-intuitif, est cohérent avec la littérature (compétitions M de Makridakis) : sur des séries relativement courtes (365 observations quotidiennes), les méthodes statistiques égalent ou dépassent fréquemment les approches d'apprentissage profond, plus gourmandes en données.`));
children.push(table(["Modèle", "MAE", "RMSE", "MAPE (%)"], fcRows, [3026, 2000, 2000, 2000]));
children.push(caption("Tableau 4.1 — Comparaison des modèles de prévision (hold-out temporel)."));
children.push(img("f4_forecast.png"));
children.push(caption("Figure 4.1 — Volume réel vs prévu sur la période de test."));
children.push(P("Validation de l'hypothèse H1 : l'avantage des modèles d'apprentissage profond sur les modèles linéaires n'est pas confirmé sur ce jeu de données, en raison de la taille limitée de la série. SARIMA reste le choix le plus robuste pour une mise en production."));

children.push(H("4.2 Résultats de la prédiction d'abandon", 2));
children.push(P(`Le tableau 4.2 compare les quatre classifieurs. Le modèle ${bestAb.Modele} obtient la meilleure aire sous la courbe ROC (AUC = ${fnum(bestAb["AUC-ROC"], 3)}). Pour un usage opérationnel, on privilégie le rappel (recall) et l'AUC-ROC, le coût d'un faux négatif — un client perdu — étant supérieur à celui d'un faux positif. Le taux d'abandon de la population modélisée est de ${fnum(R.abandon_infos.taux_abandon_pct, 2)} %.`));
children.push(table(["Modèle", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC"], abRows,
  [2426, 1320, 1320, 1320, 1320, 1320]));
children.push(caption("Tableau 4.2 — Comparaison des modèles de classification de l'abandon."));
children.push(img("f5_roc.png", 360));
children.push(caption("Figure 4.2 — Courbes ROC des quatre classifieurs."));
if (R.shap_top && R.shap_top.length) {
  children.push(P(`L'analyse SHAP (figure 4.3) identifie la charge offerte (congestion du créneau) comme le déterminant dominant de l'abandon, devant la priorité de l'appel et la durée de navigation dans le serveur vocal. Ce résultat valide l'hypothèse H3 dans sa dimension de congestion : l'abandon est avant tout un phénomène de saturation du système, ce qui légitime l'action par le dimensionnement.`));
  children.push(img("f6_shap.png"));
  children.push(caption(`Figure 4.3 — Importance SHAP des variables (${R.shap_model}).`));
}

children.push(H("4.3 Résultats de l'analyse de survie de la patience", 2));
const sv = R.survie;
children.push(P(`L'analyse de survie révèle une patience médiane des appels abandonnés de ${fnum(sv.patience_mediane_abandon_s, 0)} secondes : ${fnum(sv.abandon_moins_10s_pct, 1)} % des abandons surviennent dans les dix premières secondes et ${fnum(sv.abandon_moins_30s_pct, 1)} % dans les trente premières. La patience varie fortement selon le segment (tableau 4.3 et figure 4.4) : les prospects sont les clients les moins patients, tandis que les services courants tolèrent des attentes nettement plus longues.`));
children.push(table(["Type de service", "Patience médiane (s)"], kmRows, [5026, 4000]));
children.push(caption("Tableau 4.3 — Patience médiane estimée par type de service (Kaplan-Meier)."));
children.push(img("f7_km.png"));
children.push(caption("Figure 4.4 — Courbes de survie de la patience par type de service."));
children.push(P(`Le modèle de Cox (indice de concordance = ${fnum(R.cox_metrics.concordance, 3)}) quantifie les facteurs de risque (tableau 4.4, figure 4.5). La charge offerte présente le hazard ratio le plus élevé : une congestion accrue augmente significativement le risque instantané d'abandon, confirmant le rôle central de la saturation.`));
children.push(table(["Variable", "Hazard Ratio", "IC 95 %", "p-value"], coxRows, [3026, 2000, 2000, 2000]));
children.push(caption("Tableau 4.4 — Facteurs de risque d'abandon (modèle de Cox)."));
children.push(img("f8_cox.png"));
children.push(caption("Figure 4.5 — Hazard ratios des facteurs d'abandon (échelle logarithmique)."));

children.push(H("4.4 Dimensionnement et scénario de planification", 2));
const stf = R.staffing;
children.push(P(`En traduisant le profil de charge moyen en besoin d'agents (AHT = ${fnum(stf.aht, 0)} s, objectif de niveau de service de 80 % des appels répondus en moins de 20 secondes), le moteur Erlang C requiert jusqu'à ${fnum(stf.pic_erlangC)} agents en pic. La correction Erlang A, qui tient compte de l'impatience des clients, ramène ce pic à ${fnum(stf.pic_erlangA)} agents — illustrant que l'ignorance de l'abandon conduit à un surdimensionnement. Le besoin total s'établit autour de ${fnum(stf.agents_heures)} agents-heures par jour.`));
children.push(img("f9_staffing.png"));
children.push(caption("Figure 4.6 — Plan d'effectifs requis par créneau (Erlang C vs Erlang A)."));
children.push(P(`Mis en regard du niveau de service réel observé (${fnum(G.service_level_pct, 1)} %, très en deçà de la cible de 80 %), ce résultat objective un sous-dimensionnement structurel aux heures de pointe comme cause première de l'abandon.`));

children.push(H("4.5 Recommandations managériales", 2));
[
  "Piloter par la prévision : substituer un forecast glissant à 7 jours, par type de service, à la planification empirique, réactualisé quotidiennement (modèle SARIMA en production).",
  "Dimensionner avec Erlang A plutôt qu'Erlang C : compte tenu d'un abandon proche de 20 %, Erlang C surestime le besoin en agents ; intégrer l'impatience réduit le coût salarial à qualité de service égale.",
  "Cibler les segments impatients : router en priorité les prospects et les appels à faible patience, identifiés par l'analyse de survie, pour limiter la perte commerciale.",
  "Déployer des alertes d'abandon en temps réel : déclencher un renfort ou un rappel automatique (file d'attente virtuelle) lorsque la probabilité d'abandon prédite dépasse un seuil calibré, en priorité sur les créneaux à forte congestion.",
  "Lisser la charge : décaler les activités non urgentes (rappels sortants) hors des créneaux de pointe révélés par la heatmap.",
  "Coaching ciblé des agents : exploiter la segmentation de performance pour diffuser les bonnes pratiques des agents les plus productifs et accompagner les profils en difficulté.",
].forEach(t => children.push(bullet(t)));

children.push(H("4.6 Limites et perspectives", 2));
children.push(P("Le corpus, daté de 1999, ne reflète pas les canaux numériques contemporains (chat, e-mail, réseaux sociaux) ; l'extension à un centre multicanal constitue une perspective naturelle. La prévision gagnerait à intégrer des variables exogènes (campagnes marketing, jours fériés, événements). Enfin, la mise en production supposerait un dispositif de surveillance de la dérive des modèles (concept drift) et un ré-entraînement périodique, dans une logique MLOps."));

children.push(new Paragraph({ spacing: { before: 300 }, children: [
  new TextRun({ text: "Note méthodologique. ", bold: true, size: 18 }),
  new TextRun({ text: "L'ensemble des chiffres et figures de ce chapitre est généré automatiquement à partir des données réelles par l'application analytique développée pour ce mémoire (modules Python : ETL, KPI, prévision, classification, survie, dimensionnement). Références principales : Brown et al. (2005), JASA ; Gans, Koole & Mandelbaum (2003), M&SOM ; Chen & Guestrin (2016), KDD ; Lundberg & Lee (2017), NeurIPS ; Hyndman & Athanasopoulos (2021), FPP3.", italics: true, size: 18, color: "555555" }),
]}));

// ====================================================================
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Calibri", color: NAVY },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Calibri", color: "2E5A88" },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: "Calibri" },
        paragraph: { spacing: { before: 160, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "puces", levels: [{ level: 0, format: LevelFormat.BULLET,
    text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({
      alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: NAVY, space: 2 } },
      children: [new TextRun({ text: "Mémoire — BI & Machine Learning appliqués au centre d'appel", size: 16, color: "777777" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Page ", size: 16 }), new TextRun({ children: [PageNumber.CURRENT], size: 16 }),
        new TextRun({ text: " / ", size: 16 }), new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16 })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(DIR, "Rapport_Chapitres_III_IV.docx");
  fs.writeFileSync(out, buf);
  console.log("DOCX ecrit:", out, "(" + Math.round(buf.length / 1024) + " Ko)");
});
