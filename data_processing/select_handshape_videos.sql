-- Run from the asl_summer26 directory, with the DuckDB CLI installed:
--   duckdb :memory: < data_processing/select_handshape_videos.sql

CREATE TEMP VIEW handshape_candidates AS
SELECT
    video_id,
    label AS word,
    "Handshape" AS handshape,
    signer_id,
    split
FROM read_csv_auto(
    'data_processing/semlex.csv',
    header = true,
    all_varchar = true
)
WHERE label_type = 'asllex'
  AND "Handshape" IN ('c', 'f', 'g', 'w');

SELECT
    handshape,
    split,
    COUNT(*) AS video_count,
    COUNT(DISTINCT signer_id) AS signer_count,
    COUNT(DISTINCT word) AS word_count
FROM handshape_candidates
GROUP BY handshape, split
ORDER BY handshape, split;

SELECT video_id, word, handshape, signer_id, split
FROM handshape_candidates
ORDER BY handshape, split, word, video_id;
