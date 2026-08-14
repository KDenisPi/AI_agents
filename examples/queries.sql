-- Example queries against the DuckDB database built by load_duckdb.py.
--
-- Run the whole file, or open the database and paste one query at a time:
--
--   duckdb wiki.duckdb < queries.sql
--   duckdb wiki.duckdb
--
-- The Python library in /home/denis/projects/venv3.11 opens the same file
-- (both are v1.5.5, the storage format is tied to the version):
--
--   >>> import duckdb
--   >>> duckdb.connect('wiki.duckdb').sql("<paste a query>").show()
--
-- Tables: items, events, attributes, item_classes, sites, value_items,
-- classes, properties - the model is in db_model.sql. Queries 1 and 2 run on
-- any build; the rest need attributes and item_classes, which only exist after
-- a run made with --attrs.


-- 1. What happened in a given year in music and science.
--    Items are matched through every class they belong to, and classes carry
--    the domain tag assigned by build_class_closure.py.
SELECT i.label,
       i.description,
       c.label   AS class,
       c.domains AS area,
       p.label   AS date_type,
       e.year
FROM events e
JOIN items i        USING (qid)
JOIN item_classes l USING (qid)
JOIN classes c      ON c.qid = l."class"
JOIN properties p   ON p.pid = e.property
WHERE e.year = 1875
  AND (c.domains LIKE '%music%' OR c.domains LIKE '%science%')
ORDER BY c.domains, i.label;


-- 2. The same question asked over a whole century, counted by area.
SELECT (e.year / 100)::BIGINT * 100 AS century,
       c.domains AS area,
       count(DISTINCT e.qid) AS items
FROM events e
JOIN item_classes l USING (qid)
JOIN classes c      ON c.qid = l."class"
WHERE e.year BETWEEN 1800 AND 1899
GROUP BY 1, 2
ORDER BY items DESC;


-- 3. Which scientists lived in Germany in the 1400s.
--    "Lived in" is read as born or died inside the window, which also keeps
--    people whose other date is unknown.
WITH lifespan AS (
    SELECT qid,
           max(year) FILTER (WHERE property = 'P569') AS born,
           max(year) FILTER (WHERE property = 'P570') AS died
    FROM events
    GROUP BY qid
),
occupation AS (
    SELECT a.qid, v.label AS occupation
    FROM attributes a
    JOIN value_items v ON v.qid = a.value
    WHERE a.property = 'P106'
),
citizenship AS (
    SELECT a.qid, v.label AS country
    FROM attributes a
    JOIN value_items v ON v.qid = a.value
    WHERE a.property = 'P27'
)
-- a person usually has several occupations and sometimes several
-- citizenships, so collapse them instead of repeating the person per row
SELECT i.label,
       string_agg(DISTINCT o.occupation, ', ') AS occupations,
       string_agg(DISTINCT z.country, ', ')    AS countries,
       min(l.born) AS born,
       min(l.died) AS died
FROM lifespan l
JOIN items i      USING (qid)
JOIN occupation o USING (qid)
JOIN citizenship z USING (qid)
-- word-boundary match so a country name isn't caught as a substring of another
WHERE regexp_matches(z.country, '(?i)\bGerman')
  -- until occupations are mapped to areas, name the scientific ones directly
  AND o.occupation IN ('physicist', 'chemist', 'astronomer', 'mathematician',
                       'naturalist', 'physician', 'alchemist', 'botanist',
                       'geographer', 'cartographer', 'engineer')
  AND (l.born BETWEEN 1400 AND 1499 OR l.died BETWEEN 1400 AND 1499)
GROUP BY i.label
ORDER BY born;


-- 4. Anyone by occupation and period, the generic form of query 3.
--    Change the two parameters at the top and the rest follows.
WITH params AS (SELECT 'composer' AS want_occupation, 1700 AS from_year, 1799 AS to_year),
lifespan AS (
    SELECT qid,
           max(year) FILTER (WHERE property = 'P569') AS born,
           max(year) FILTER (WHERE property = 'P570') AS died
    FROM events
    GROUP BY qid
)
SELECT i.label, v.label AS occupation, l.born, l.died
FROM attributes a
JOIN value_items v ON v.qid = a.value
JOIN items i       ON i.qid = a.qid
JOIN lifespan l    ON l.qid = a.qid
CROSS JOIN params
WHERE a.property = 'P106'
  AND v.label = params.want_occupation
  AND l.born BETWEEN params.from_year AND params.to_year
ORDER BY l.born;


-- 5. Musical works of a year with the people and genre attached to them.
SELECT i.label AS work,
       max(CASE WHEN a.property = 'P86'  THEN v.label END) AS composer,
       max(CASE WHEN a.property = 'P175' THEN v.label END) AS performer,
       max(CASE WHEN a.property = 'P136' THEN v.label END) AS genre,
       min(e.year) AS year
FROM events e
JOIN items i        USING (qid)
JOIN item_classes l USING (qid)
JOIN classes c      ON c.qid = l."class"
LEFT JOIN attributes a  ON a.qid = e.qid AND a.property IN ('P86', 'P175', 'P136')
LEFT JOIN value_items v ON v.qid = a.value
WHERE e.year = 1875
  AND c.domains LIKE '%music%'
GROUP BY i.label
ORDER BY i.label;


-- 6. Events of a period with where they happened and who took part.
SELECT i.label AS event,
       min(e.year) AS year,
       max(CASE WHEN a.property = 'P17'  THEN v.label END) AS country,
       max(CASE WHEN a.property = 'P276' THEN v.label END) AS location,
       string_agg(DISTINCT CASE WHEN a.property = 'P710' THEN v.label END, ', ') AS participants
FROM events e
JOIN items i        USING (qid)
JOIN item_classes l USING (qid)
JOIN classes c      ON c.qid = l."class"
LEFT JOIN attributes a  ON a.qid = e.qid AND a.property IN ('P17', 'P276', 'P710')
LEFT JOIN value_items v ON v.qid = a.value
WHERE e.year BETWEEN 1600 AND 1699
  AND c.domains LIKE '%event%'
GROUP BY i.label
ORDER BY year
LIMIT 50;


-- 7. Which occupations actually occur in the data.
--    This is the list to map onto areas (science, music, art) - far smaller
--    than the full occupation vocabulary, because only these are used.
SELECT v.label AS occupation, count(*) AS people
FROM attributes a
JOIN value_items v ON v.qid = a.value
WHERE a.property = 'P106'
GROUP BY 1
ORDER BY people DESC
LIMIT 200;


-- 8. Data quality.
--    Date precision: month = 0 means the source only knew the year.
SELECT p.label AS date_type,
       count(*) AS dates,
       count(*) FILTER (WHERE e.month = 0) AS year_only,
       count(*) FILTER (WHERE e.bc)        AS bc,
       min(e.year) AS earliest,
       max(e.year) AS latest
FROM events e
JOIN properties p ON p.pid = e.property
GROUP BY 1
ORDER BY dates DESC;

--    Attribute values that resolve_values.py could not label. Anything here
--    means the values file is stale against the extract.
SELECT a.property, count(*) AS unresolved
FROM attributes a
LEFT JOIN value_items v ON v.qid = a.value
WHERE v.qid IS NULL
GROUP BY 1
ORDER BY unresolved DESC;


-- 9. A single named work's date, disambiguated by its creator.
--    The stage-1 "point lookup" shape: filters.name = the work, related_entity
--    {creates, <person>} to pick the right one, answer_field = "time" -> project
--    the year instead of returning the matching rows. P86 is composer, P50 author;
--    match the work by label and the creator through their value_items label.
SELECT i.label      AS work,
       comp.label   AS creator,
       min(e.year)  AS year
FROM items i
JOIN attributes a     ON a.qid = i.qid AND a.property IN ('P86', 'P50')
JOIN value_items comp ON comp.qid = a.value
LEFT JOIN events e     ON e.qid = i.qid
WHERE i.label ILIKE '%Symphony%5%'
  AND comp.label ILIKE '%Beethoven%'
GROUP BY i.label, comp.label;


-- 10. People of an occupation whose lifespan overlaps a named person's.
--     The stage-1 time_constraint {relative_to_entity, overlap, <anchor>} shape:
--     resolve the anchor's birth/death years first, then keep candidates whose
--     own lifespan intersects it. Occupation is attribute P106 (attributes ->
--     value_items), NOT a word in the item label. P569 = date of birth,
--     P570 = date of death. For "before" use l.died <= anchor.born; for "after"
--     use l.born >= anchor.died.
WITH lifespan AS (
    SELECT qid,
           max(year) FILTER (WHERE property = 'P569') AS born,
           max(year) FILTER (WHERE property = 'P570') AS died
    FROM events
    GROUP BY qid
),
-- Resolve the anchor to a real dated person. Many items share the surname
-- (a whole van Beethoven family, plus craters/films/works), so require a birth
-- year and pick the most-documented match - a stable proxy for "the famous one"
-- - instead of an arbitrary LIMIT 1 on the bare label.
anchor AS (
    SELECT l.born, l.died
    FROM items i
    JOIN lifespan l USING (qid)
    WHERE i.label ILIKE '%Beethoven%'
      AND l.born IS NOT NULL
    ORDER BY (SELECT count(*) FROM attributes a WHERE a.qid = i.qid) DESC
    LIMIT 1
)
SELECT i.label, l.born, l.died
FROM attributes a
JOIN value_items v ON v.qid = a.value
JOIN items i       ON i.qid = a.qid
JOIN lifespan l    ON l.qid = a.qid
WHERE a.property = 'P106'          -- occupation
  AND v.label = 'composer'
  -- interval overlap: born on/before the anchor died, died on/after it was born.
  -- This covers all three cases: born within, died within, and lifespans that
  -- enclose the anchor's entirely (born earlier and died later). Reference the
  -- single-row anchor as scalar subqueries so it needs no join and can't be
  -- dropped when the rest of the query is restructured (adding more filters).
  AND l.born <= (SELECT died FROM anchor)
  AND l.died >= (SELECT born FROM anchor)
  -- sanity guard: drop impossible lifespans from bad death-years in the data
  -- (e.g. a 1533 birth paired with a stray 1985 death claim)
  AND l.died - l.born <= 110
ORDER BY l.born;
