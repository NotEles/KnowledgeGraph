import argparse
import json
from pathlib import Path

import pandas as pd


class TMDBExtractor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    @staticmethod
    def _parse_json_list(value):
        if pd.isna(value) or value == "":
            return []
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []

    def extract_all(self, limit=None):
        all_triples = []
        rows = self.df.head(limit) if limit is not None else self.df

        for _, row in rows.iterrows():
            movie_title = row.get("title") or row.get("original_title")
            if not movie_title:
                continue

            for genre in self._parse_json_list(row.get("genres")):
                genre_name = genre.get("name")
                if genre_name:
                    all_triples.append({
                        "head": movie_title,
                        "head_label": "Movie",
                        "rel": "HAS_GENRE",
                        "tail": genre_name,
                        "tail_label": "Genre",
                    })

            for keyword in self._parse_json_list(row.get("keywords")):
                keyword_name = keyword.get("name")
                if keyword_name:
                    all_triples.append({
                        "head": movie_title,
                        "head_label": "Movie",
                        "rel": "HAS_KEYWORD",
                        "tail": keyword_name,
                        "tail_label": "Keyword",
                    })

            for company in self._parse_json_list(row.get("production_companies")):
                company_name = company.get("name")
                if company_name:
                    all_triples.append({
                        "head": movie_title,
                        "head_label": "Movie",
                        "rel": "PRODUCED_BY",
                        "tail": company_name,
                        "tail_label": "ProductionCompany",
                    })

            for country in self._parse_json_list(row.get("production_countries")):
                country_name = country.get("name")
                if country_name:
                    all_triples.append({
                        "head": movie_title,
                        "head_label": "Movie",
                        "rel": "PRODUCED_IN",
                        "tail": country_name,
                        "tail_label": "Country",
                    })

            for language in self._parse_json_list(row.get("spoken_languages")):
                language_name = language.get("name")
                if language_name:
                    all_triples.append({
                        "head": movie_title,
                        "head_label": "Movie",
                        "rel": "SPOKEN_IN",
                        "tail": language_name,
                        "tail_label": "Language",
                    })

        return all_triples


def main():
    from graph_builder import KGBuilder

    default_csv = Path(__file__).resolve().parent.parent / "sample" / "tmdb" / "tmdb_5000_movies.csv"
    parser = argparse.ArgumentParser(description="Extract TMDB triples and load a small sample into Neo4j.")
    parser.add_argument(
        "--csv",
        default=str(default_csv),
        help="Path to tmdb_5000_movies.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of rows from the top of the CSV to process",
    )
    parser.add_argument("--uri", default="bolt://localhost:7688", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j user")
    parser.add_argument("--password", default="your_new_password", help="Neo4j password")
    args = parser.parse_args()

    extractor = TMDBExtractor(args.csv)
    triples = extractor.extract_all(limit=args.limit)

    kg = KGBuilder(args.uri, args.user, args.password)
    try:
        print(f"开始导入前 {args.limit} 行数据，共 {len(triples)} 条关系。")
        for triple in triples:
            kg.add_relation(
                triple["head"],
                triple["head_label"],
                triple["rel"],
                triple["tail"],
                triple["tail_label"],
            )
            print(f"已插入: {triple['head']} -> {triple['rel']} -> {triple['tail']}")
    finally:
        kg.close()


if __name__ == "__main__":
    main()