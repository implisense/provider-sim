# Report: Das Soja-Lieferketten-Szenario

**Datei:** `s1-soja_icio.pdl.yaml`
**Erstellt:** April 2026
**Zielgruppe:** Nicht-technische Leser ohne volkswirtschaftlichen Hintergrund

---

## Was ist das hier eigentlich?

Diese Datei ist das digitale Herzstück einer **Computer-Simulation**, die zeigt, was passiert, wenn die Versorgung Europas mit Soja ins Stocken gerät. Sie beschreibt eine Art „Spielfeld" für einen KI-gesteuerten Krisentest: Auf der einen Seite versucht ein simulierter Angreifer, die Versorgung zu stören – auf der anderen Seite versucht ein Verteidiger, Schäden abzuwenden.

Das Szenario trägt den Titel **„Soja-Futtermittel-Lieferkette"** und ist das erste von neun solcher Szenarien im Forschungsprojekt PROVIDER.

---

## Warum ausgerechnet Soja?

Soja ist eines der wichtigsten Grundnahrungsmittel – nicht für uns direkt, sondern für die Tiere, die wir essen. Rund **70–80 % der Sojabohnen weltweit werden als Tierfutter** verwendet. Ohne Soja kein billiges Hähnchen, kein günstiges Schweinefleisch, keine erschwinglichen Milchprodukte.

Das Problem: Europa baut kaum selbst Soja an. Wir importieren fast alles – und zwar vor allem aus **Brasilien**. Das macht unsere Lebensmittelversorgung anfällig für Ereignisse, die Tausende Kilometer entfernt stattfinden.

---

## Wer ist beteiligt? Die 20 Akteure des Szenarios

Das Szenario kennt 20 Akteure („Entities"), die alle miteinander verbunden sind:

### Anbauländer (Lieferanten)

| Akteur | Land | Bedeutung |
|---|---|---|
| Sojaanbau Brasilien | Brasilien | Wichtigster Lieferant: ~62 Mrd. USD Exportwert pro Jahr |
| Sojaanbau Argentinien | Argentinien | Zweitgrößter Lieferant, aber nur ~3,3 Mrd. USD Sojabohnen |
| Sojaanbau USA | USA | Notfall-Lieferant: ~34 Mrd. USD, weltweit größter Exporteur |

### Häfen (Drehscheiben des Welthandels)

| Akteur | Land | Besonderheit |
|---|---|---|
| Hafen Santos | Brasilien | Größter Soja-Exporthafen der Welt – 30 % aller brasilianischen Ausfuhren |
| Hafen Paranaguá | Brasilien | Zweitgrößter brasilianischer Soja-Hafen |
| US-Golfhäfen (New Orleans, Houston) | USA | Ausweichroute bei Brasilien-Ausfällen |
| Hafen Rotterdam | Niederlande | Wichtigstes Einfallstor für Soja in Europa |
| Hafen Hamburg | Deutschland | Zweitwichtigster europäischer Soja-Importhafen |

### Verarbeitung & Produktion in der EU

| Akteur | Beschreibung |
|---|---|
| EU Ölmühlen | Pressen Sojabohnen zu Sojaschrot (Tierfutter) und Sojaöl |
| Futtermittelhersteller | Mischen Sojaschrot zu fertigem Tierfutter |
| Geflügelbetriebe | Verbrauchen ~60 % des Futters; besonders abhängig von Sojaprotein |
| Schweinebetriebe | Zweitgrößter Verbraucher von Sojaschrot |
| Milchwirtschaft | Einsatz von Sojaschrot in Kraftfutter für Rinder |

### Endverbraucher & Märkte

| Akteur | Beschreibung |
|---|---|
| Lebensmitteleinzelhandel | Supermarkt & Co. – gibt Preiserhöhungen an Verbraucher weiter |
| Verbraucher | Wir alle – am Ende der Kette |
| Sojaöl-Markt | Nebenprodukt der Ölmühlen (~20 % der Sojabohne wird zu Öl) |
| Alternative Proteinquellen | Rapsschrot, Sonnenblumenschrot, Erbsenprotein als Ersatz für Sojaschrot |

### Unterstützende Systeme

| Akteur | Beschreibung |
|---|---|
| Erdgasversorgung | Ölmühlen brauchen viel Energie – ein Gaspreisschock trifft sie direkt |
| Düngemittelversorgung | Brasilianische Bauern brauchen Dünger – Ausfall trifft den Anbau |
| Industrielle Futtermittel-Lagerbestände | Puffer für ca. 30–45 Tage Verbrauch |
| Strategische Reserven | Schnell einsetzbar, aber begrenzt |

---

## Wie hängt alles zusammen? Der Weg des Sojas

```
Brasilianische Felder
       ↓
  Hafen Santos  ─(Ausweich)→  Hafen Paranaguá
       ↓                              ↓
  Hafen Rotterdam  ←────────────────┘
       ↓
  EU Ölmühlen  ──── (Nebenprodukt) ──→  Sojaöl-Markt
       ↓
  Futtermittelhersteller
    ↙    ↓    ↘
Geflügel  Schwein  Milch
    ↘    ↓    ↙
  Lebensmittelhandel
       ↓
   Verbraucher
```

Daneben laufen zwei **Notfall-Routen**:
- **Argentinien** kann bis zu ~7 % des brasilianischen Sojabohnen-Ausfalls ersetzen (Achtung: Bei gleichzeitiger Dürre in Südamerika funktioniert das nicht!)
- **USA** können über ihre Golfhäfen einspringen und decken theoretisch bis zu ~70 % eines brasilianischen Ausfalls ab

---

## Was kann schiefgehen? Die Krisen-Ereignisse

Das Szenario simuliert **mehr als 40 Ereignisse** – von Naturkatastrophen über Marktschocks bis hin zu staatlichen Handelseingriffen. Hier die wichtigsten:

### Naturkatastrophen

| Ereignis | Wahrscheinlichkeit pro Jahr | Auswirkung |
|---|---|---|
| **Dürre in Brasilien** | 15 % | Erntemengen –40 % für 90 Tage; ausgelöst z.B. durch La Niña (tatsächlich 2021/22 passiert) |

### Infrastrukturausfälle

| Ereignis | Wahrscheinlichkeit pro Jahr | Auswirkung |
|---|---|---|
| **Santos-Hafen-Ausfall** (70 % Kapazitätsverlust) | 8 % | Versorgungseinbruch –21 %; wirtschaftlicher Schaden weltweit: ~24–29 Mrd. USD |
| **Hafenstau Santos** | 25 % | –20 % Kapazität für 30 Tage; tatsächlich 2022: 88 Schiffe in der Warteschlange |
| **Ölmühlen-Drosselung** (EU) | – | –30 % Kapazität für 60 Tage; folgt auf Gaspreis-Explosion |

### Marktschocks (Kettenreaktionen)

Diese Ereignisse werden durch andere ausgelöst – sie zeigen, wie sich eine Krise durch die Kette fortpflanzt:

| Ereignis | Auslöser | Auswirkung |
|---|---|---|
| **Soja-Exportrückgang** | Dürre in Brasilien | Exportmenge –12 % für 120 Tage |
| **Futtermittelpreisanstieg** | Exportrückgang | Preise **+45 %** für 180 Tage (2022 real: +31–50 %) |
| **Sojaschrot-Engpass** | Santos-Ausfall | Preise **+35 %**, Versorgung –21 % für 180 Tage |
| **Druck auf Tierproduktion** | Futtermittelpreise | Fleischerzeugung –9 %, Preise +20 % (2022 real: Schwein –9,6 %) |
| **Verbraucherpreiserhöhung** | Druck auf Tierproduktion | Fleischpreise **+18 %** (2022 real: Nahrungsmittel +13,4 %) |
| **Gaspreis-Explosion** | Eigenständig (20 % p.a.) | Gaspreise **+200 %** für 180 Tage – trifft Ölmühlen und Düngemittel |
| **Ammoniak-Produktionsstopp** | Gaspreis-Explosion | Düngemittel –70 % für 90 Tage (real: BASF/Yara 2022) |
| **Sojaöl-Engpass** | Ölmühlen-Drosselung | Sojaöl –25 %, Preise +60 % (real 2022: +85 %) |

### Staatliche Handelseingriffe (Regulatorische Ereignisse)

Ein besonderer Aspekt dieses Szenarios: Es enthält **über 25 reale Handelspolitik-Maßnahmen** aus dem Global Trade Alert (GTA) – einer Datenbank tatsächlich bestehender Handelseingriffe:

**Argentinien** hat zahlreiche Export-Beschränkungen auf Agrarrohstoffe:
- Exportlizenzen für Getreide, Ölsaaten, Sojaschrot, Sojaöl
- Exportquoten (bis zu –15 % Versorgung)
- Exportsteuern (bis zu –10 % Versorgung)

**USA** haben ebenfalls Eingriffe:
- Importzölle auf Sojabohnen (wirken paradoxerweise als leichter Angebotsschub für EU-Käufer)
- Exportlizenzpflichten für bestimmte Produktgruppen

Diese regulatorischen Ereignisse treten mit einer Wahrscheinlichkeit von 25–50 % pro Jahr ein – sie sind also keine Ausnahmen, sondern Teil des normalen Handelsgeschehens.

---

## Was passiert, wenn es eng wird? Die Ausweichmöglichkeiten

Das Szenario modelliert **15 Ausweich- und Ersatzstrategien** (Substitutionen). Hier die wichtigsten:

### Ersatzlieferanten für Soja

| Strategie | Deckung | Anlaufzeit | Haken |
|---|---|---|---|
| Mehr Soja aus Argentinien | 7 % des BR-Ausfalls | 14 Tage | Bei La-Niña-Dürre ebenfalls betroffen |
| Mehr Soja aus USA | 70 % des BR-Ausfalls | 30 Tage | Teurer (+25 %), erst nach 30 Tagen aktiv |

### Ersatzproteine für Tierfutter

Bei Sojaschrot-Engpass können Futtermittelhersteller auf andere Eiweißquellen umsteigen:

| Ersatzstoff | Deckungspotenzial | Qualität | Anlaufzeit |
|---|---|---|---|
| Fischmehl | 24 % | etwas besser | 60 Tage |
| Rapsschrot (00-Qualität) | 14 % | etwas schlechter | 60 Tage |
| Sonnenblumenschrot | 13 % | merklich schlechter | 60 Tage |
| Fleisch-/Knochenmehl | 16 % | leicht schlechter | 60 Tage |
| Palmkernschrot | 2 % | deutlich schlechter | 60 Tage |

**Wichtig:** Diese Alternativen summieren sich auf rund 70 % – ein vollständiger Ersatz ist nicht möglich. Deutschland ist besonders verwundbar: Nur **19,5 %** des deutschen Sojaschrot-Importausfalls können durch alternative Lieferanten aus USA, Argentinien, Ukraine und Kanada gedeckt werden.

### Verbraucherverhalten

Wenn Fleisch teurer wird, reagieren Verbraucher:
- **–8 % Fleischnachfrage** bei mehr als 15 % Preisanstieg (Umstieg auf pflanzliche Alternativen)
- 2022 real gemessen: –4,2 % Fleischkonsum in Deutschland (DGE)

---

## Die globalen Dominoeffekte

Dank **volkswirtschaftlicher Input-Output-Analyse** (OECD ICIO 2025) zeigt das Szenario, was ein Santos-Hafenausfall für die Weltwirtschaft bedeutet:

> Ein 70-prozentiger Ausfall des Hafens Santos für 90 Tage löst wirtschaftliche Schäden von **24–29 Milliarden US-Dollar** weltweit aus – der Schaden ist also fast **doppelt so groß** wie der direkte Handelswert des betroffenen Sojas (13 Mrd. USD). Das nennt man den „Leontief-Multiplikator-Effekt": Jeder Dollar weniger Soja-Export zieht weitere 86 Cent an indirekten Schäden nach sich.

Die am stärksten betroffenen Länder und Sektoren bei einem Santos-Ausfall:
1. **China, Lebensmittelverarbeitung**: –4,9 Mrd. USD
2. **China, Landwirtschaft**: –4,7 Mrd. USD
3. **Rest der Welt, Landwirtschaft**: –0,5 Mrd. USD

Das zeigt: Eine Krise in einem brasilianischen Hafen trifft zuerst China – und über Rückkopplungen auch Europa.

---

## Was macht die Simulation damit?

Das Szenario läuft als Computer-Experiment über 365 simulierte Tage. Dabei:

1. **Jeder simulierte Tag** berechnet neu, wie viel Soja noch fließt, wie hoch die Preise sind, wie gesund jedes Kettenglied ist.
2. **Ein KI-Angreifer** versucht, kritische Punkte zu treffen – zum Beispiel Santos oder die EU-Ölmühlen.
3. **Ein KI-Verteidiger** versucht, Puffer zu stärken und Schäden zu begrenzen.
4. **Nach vielen solchen Durchläufen** lernen beide KIs dazu – der Angreifer, wo die Kette am schwächsten ist; der Verteidiger, wie man Krisen am besten abfedert.

Das Ergebnis: Aussagen darüber, **welche Punkte in der Lieferkette besonders kritisch** sind und welche Maßnahmen die Versorgungssicherheit am stärksten verbessern.

---

## Die wichtigsten Erkenntnisse auf einen Blick

| Befund | Bedeutung |
|---|---|
| **Brasilien dominiert** | ~62 Mrd. USD Soja-Exporte; kein anderes Land ist annähernd so wichtig |
| **Santos ist der Flaschenhals** | 30 % aller brasilianischen Soja-Ausfuhren laufen durch einen einzigen Hafen |
| **Deutschland ist besonders exponiert** | Nur ~20 % Substitutionsdeckung bei Sojaschrot-Ausfall |
| **Gaspreise verbinden zwei Krisen** | Eine Energiekrise trifft gleichzeitig Düngemittel (Ernte) und Ölmühlen (Verarbeitung) |
| **Dominoeffekte sind der eigentliche Risikotreiber** | Der Wirtschaftsschaden ist ~2× größer als der direkte Handelsverlust |
| **Argentinien ist kein zuverlässiger Notfalllieferant** | Bei südamerikanischer Dürre sind beide Länder gleichzeitig betroffen |
| **USA sind die eigentliche Sicherheitsreserve** | Können theoretisch 70 % eines Brasilien-Ausfalls decken – aber mit 30 Tagen Anlaufzeit und +25 % Kosten |

---

## Fazit

Das Soja-Szenario zeigt modellhaft, wie **globale Lieferketten eine versteckte Fragilität** besitzen: Alles läuft reibungslos, solange nichts passiert. Sobald aber ein Ereignis eintritt – sei es eine Dürre, ein Hafenausfall oder eine Handelspolitik-Entscheidung in Buenos Aires – kann sich das durch die gesamte Kette bis auf den Preis von Hähnchenbrust im deutschen Supermarkt auswirken.

Die Simulation hilft dabei, diese Schwachstellen **vor einer echten Krise** zu identifizieren – damit Politik, Unternehmen und Behörden gezielt Vorsorge treffen können.

---

*Report basiert auf Simulationsdaten aus `s1-soja_icio.pdl.yaml` (PROVIDER-Projekt, OECD ICIO 2025 Edition, BACI 2022, CoyPu Knowledge Graph, Global Trade Alert)*
