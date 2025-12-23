# для шага 3 используется: Aspect-based analysis (baseline)

from aspect_analysis_3.aspect_analysis_pipeline import (
    AspectAnalysisPipeline,
    AspectAnalysisConfig,
)


def main():
    config = AspectAnalysisConfig(
        input_sentence_file=(
            "topic_sentiment_2/topic_sentiment_results/"
            "sentence_topics_with_sentiment.csv"
        ),
        min_aspect_length=3,
        min_aspect_count=10,
    )

    pipeline = AspectAnalysisPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
