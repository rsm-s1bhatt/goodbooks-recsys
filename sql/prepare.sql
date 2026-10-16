-- Build clean interaction tables from the raw goodbooks-10k CSVs.
CREATE OR REPLACE TABLE ratings AS
SELECT user_id::INT AS user_id, book_id::INT AS book_id, rating::INT AS rating
FROM read_csv_auto('data/raw/ratings.csv')
QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id, book_id ORDER BY rating DESC) = 1;  -- drop duplicate pairs

CREATE OR REPLACE TABLE books AS
SELECT book_id::INT AS book_id, title, authors, original_publication_year AS pub_year
FROM read_csv_auto('data/raw/books.csv');

-- Tag text per book, weighted by how often readers applied each tag (top 30 tags per book).
CREATE OR REPLACE TABLE book_tag_text AS
WITH bt AS (
  SELECT b.book_id, t.tag_name, bt.count AS n,
         ROW_NUMBER() OVER (PARTITION BY b.book_id ORDER BY bt.count DESC) AS rk
  FROM read_csv_auto('data/raw/book_tags.csv') bt
  JOIN read_csv_auto('data/raw/books.csv') b ON b.goodreads_book_id = bt.goodreads_book_id
  JOIN read_csv_auto('data/raw/tags.csv') t ON t.tag_id = bt.tag_id
  WHERE bt.count > 0
)
SELECT book_id, string_agg(replace(tag_name, ' ', '_'), ' ') AS tags
FROM bt WHERE rk <= 30 GROUP BY book_id;
