# Planner

Ideas are suggestions, swap freely. Effort: **Quick** (under 30 min) · **Standard** (about an hour) · **Punch** (hero map, when you have time).
Tool picks keep a few days for **Mapbox** (2026 sponsor) and **ArcGIS Pro** (tag Esri).

| Day | Date | Prompt | Study | Idea | Effort | Tool | Status |
|---|---|---|---|---|---|---|---|
| 1 | Nov 1 | Points | A | Where AI Gets Built: every US data center + the new AI campuses | Standard | Mapbox | Posted |
| 2 | Nov 2 | Lines | City | Big Sur Marathon: the course in plan and section, in three line weights | Quick | Python static | Building |
| 3 | Nov 3 | Polygons | City | What San Francisco Lets You Build: official zoning districts (DataSF) | Quick | Python static | Building |
| 4 | Nov 4 | Clusters | City | Where New York Goes for Indian: DBSCAN on Indian restaurants (NYC health department) | Quick | Python static | Building |
| 5 | Nov 5 | Sight | City | What Billy Penn Can No Longer See: horizon blocked by towers taller than his hat | Standard | Python static | Building |
| 6 | Nov 6 | Vintage | City | Bombay Was Seven Islands: the 1843 map with today's coastline over it | Standard | Python static | Building |
| 7 | Nov 7 | 10 minute map | City | Philadelphia Passes the 10-Minute Test: walks to parks of different sizes, with heat-vulnerable tracts outlined (City of Philadelphia, OSM streets) | Standard | Python static | Building |
| 8 | Nov 8 | Utopia | D | Where data centers *should* go |  |  |  |
| 9 | Nov 9 | Urban-rural | A | Leaving the city for farmland, plus data center alleys (moved from Day 4) |  |  |  |
| 10 | Nov 10 | Prompting only |  |  | Quick |  |  |
| 11 | Nov 11 | Sound |  |  |  |  |  |
| 12 | Nov 12 | Power | C | What AI Burns: power plants + main fuel near each site (EIA) | Punch | Mapbox (+ ArcGIS Pro option) | Notebook ready |
| 13 | Nov 13 | Interactions | City |  |  |  |  |
| 14 | Nov 14 | Borgesian map | City |  |  |  |  |
| 15 | Nov 15 | Inside out | D | Where AI Is at Risk: FEMA risk by county (overall or wildfire, heat...) | Standard | Mapbox | Notebook ready |
| 16 | Nov 16 | Collaborative | City |  |  |  |  |
| 17 | Nov 17 | Light & dark | C | Night lights | Punch | Mapbox |  |
| 18 | Nov 18 | NULL | A | Counties with zero data centers | Quick |  |  |
| 19 | Nov 19 | Smell | City |  |  |  |  |
| 20 | Nov 20 | Hexagons | B | What AI Drinks: water stress in hexagons (moved from Day 3, 39% stat, notebook exists) |  |  | Notebook ready, rework to hexagons |
| 21 | Nov 21 | OpenStreetMap | City | Learning to Ski from NYC and Philly: beginner terrain at ski areas within a day trip, from OSM trail tags, with drive times from both cities | Standard | Mapbox or static | Building |
| 22 | Nov 22 | Projections |  |  | Quick | Mapbox |  |
| 23 | Nov 23 | Taste | City |  |  |  |  |
| 24 | Nov 24 | Network | A | Fiber + power lines |  |  |  |
| 25 | Nov 25 | Is this a map? |  |  |  |  |  |
| 26 | Nov 26 | Water | B | Finale: footprint + exposure | Punch | ArcGIS Pro |  |
| 27 | Nov 27 | New tool |  |  |  | Mapbox Studio? |  |
| 28 | Nov 28 | Feeling | City |  |  |  |  |
| 29 | Nov 29 | Raster | C | Extreme heat | Punch | ArcGIS Pro |  |
| 30 | Nov 30 | Pen & paper | City |  | Quick | Paper |  |

Status: *Idea → Building → Posted → Sent to MUSA*

## Studies · The Physical Footprint of AI

Each map stands alone, and also adds a column to the master table (`data/processed/datacenters_master.geojson`).
When a study has 2 or 3 maps, they get stitched into one scrolling study page for the portfolio.

| Study | Question | Maps |
|---|---|---|
| A · Where AI Gets Built | Where are data centers, and where are the new AI campuses going? | Points, Clusters, Urban-rural, NULL, Network |
| B · What AI Drinks | Whose water do they share? | Hexagons (water stress), Water |
| C · What AI Burns | What powers them, and how hot is it getting? | Power, Light & dark, Raster (heat) |
| D · Where AI Is at Risk | What climate risks do they face, and where should they go instead? | Inside out (wildfire), Utopia |
| City · City Senses | The lighter thread | Lines, Polygons, Clusters, Sight, Vintage, 10 minute map, Interactions, Borgesian, Collaborative, Smell, OSM, Taste, Feeling, Pen & paper |
