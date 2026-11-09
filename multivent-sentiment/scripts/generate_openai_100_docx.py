"""Create a shareable Word report for the finalized MULTIVENT 100-review evaluation."""
import csv
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


def run(text: str, bold: bool = False, size: int | None = None) -> str:
    props = "<w:rPr>" + ("<w:b/>" if bold else "") + (f'<w:sz w:val="{size}"/>' if size else "") + "</w:rPr>"
    parts = escape(str(text)).split("\n")
    return "".join(f"<w:r>{props}<w:t xml:space=\"preserve\">{part}</w:t></w:r>" + ("<w:r><w:br/></w:r>" if i < len(parts) - 1 else "")
                   for i, part in enumerate(parts))


def paragraph(text: str = "", style: str | None = None, bold: bool = False, size: int | None = None) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}{run(text, bold=bold, size=size)}</w:p>"


def cell(text: str, header: bool = False) -> str:
    return f'<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/></w:tcPr>{paragraph(text, bold=header, size=18)}</w:tc>'


def table(headers: list[str], rows: list[list[str]]) -> str:
    border = '<w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders>'
    grid = "".join('<w:gridCol w:w="1100"/>' for _ in headers)
    head = "<w:tr>" + "".join(cell(item, header=True) for item in headers) + "</w:tr>"
    body = "".join("<w:tr>" + "".join(cell(item) for item in row) + "</w:tr>" for row in rows)
    return f'<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>{border}</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{head}{body}</w:tbl>'


def page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def document_xml(results: dict, prediction_rows: list[dict]) -> str:
    metrics = results["metrics_by_dataset"]
    order = ["HiliSenti v1", "MAMS ACSA", "Booking.com Hotel Reviews", "UCI Yelp Sentiment Labelled Sentences"]
    audit = results["api_execution_audit"]
    overall = results["overall_metrics"]
    body = [
        paragraph("MULTIVENT — 100-Review OpenAI Sentiment Analysis Evaluation", "Title"),
        paragraph("Controlled held-out evaluation report for Chapter 4 documentation."),
        paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Model: {results['configured_model']}."),
        paragraph("Evaluation design", "Heading1"),
        paragraph("A seeded (42), stratified sample of 100 held-out reviews was evaluated: 25 records from each source, with 13 positive and 12 negative records per source. The pipeline used original-text normalization, LDA topic inference, FAISS retrieval, and the configured OpenAI structured-output provider. Positive is the positive class for precision, recall, and F1."),
        paragraph("HiliSenti records come from its published test split. MAMS, Booking.com, and UCI source imports do not have equivalent published test partitions in this project; their records come exclusively from the project's isolated held-out test partition. Text hashes, IDs, and group IDs were checked against LDA training and validation data with zero collisions."),
        paragraph("Results by dataset", "Heading1"),
        table(
            ["Dataset", "N", "Accuracy", "Positive Precision", "Positive Recall", "Positive F1", "Macro F1", "Correct", "Incorrect"],
            [[source, str(metrics[source]["total_evaluated_reviews"]), f'{metrics[source]["accuracy"]:.4f}',
              f'{metrics[source]["positive_precision"]:.4f}', f'{metrics[source]["positive_recall"]:.4f}',
              f'{metrics[source]["positive_f1_score"]:.4f}', f'{metrics[source]["macro_f1_score"]:.4f}',
              str(metrics[source]["correct_predictions"]), str(metrics[source]["incorrect_predictions"])] for source in order],
        ),
        paragraph("Overall results", "Heading1"),
        paragraph(f"Accuracy: {overall['accuracy']:.4f}. Positive precision: {overall['positive_precision']:.4f}. Positive recall: {overall['positive_recall']:.4f}. Positive F1-score: {overall['positive_f1_score']:.4f}. Macro F1-score: {overall['macro_f1_score']:.4f}. Correct: {overall['correct_predictions']}. Incorrect: {overall['incorrect_predictions']}."),
        paragraph(f"Confusion matrix, actual [negative, positive] × predicted [negative, positive]: {overall['confusion_matrix']}. This corresponds to TN=46, FP=2, FN=3, TP=49."),
        paragraph("OpenAI usage and cost audit", "Heading1"),
        paragraph(f"The 100 unique scored records used {results['evaluation_token_usage']['input_tokens']:,} input tokens and {results['evaluation_token_usage']['output_tokens']:,} output tokens ({results['evaluation_token_usage']['total_tokens']:,} total), at an evaluation-record cost of ${results['evaluation_cost_usd']:.6f} USD."),
        paragraph(f"Actual account consumption was {audit['billable_completed_requests']} completed API calls, {audit['total_api_token_usage']['input_tokens']:,} input tokens and {audit['total_api_token_usage']['output_tokens']:,} output tokens ({audit['total_api_token_usage']['total_tokens']:,} total), costing ${audit['total_api_cost_usd']:.6f} USD. Eight duplicate calls occurred during interrupted/resumed execution; their labels matched the associated scored records and are excluded from the 100-record metrics."),
        paragraph("Limitations", "Heading1"),
        paragraph("This is an automated evaluation against dataset-provided labels and is not independently human-validated. Booking.com field-position labels are weak supervision. The final results contain no failed OpenAI requests; preliminary sandbox connection failures produced no completed API response or recorded token usage and are not part of the evaluation."),
        page_break(),
        paragraph("Appendix A — Complete prediction records", "Heading1"),
        paragraph("The following records correspond to the 100 unique scored reviews. Full raw API audit records are maintained separately in raw_openai_responses.jsonl."),
    ]
    for record in prediction_rows:
        body += [
            paragraph(f"{record['Test ID']} — {record['Dataset Source']}", "Heading2"),
            paragraph("Original Review", bold=True), paragraph(record["Original Review"]),
            paragraph(
                f"Ground truth: {record['Ground Truth Label']} | Prediction: {record['Model Prediction']} | "
                f"Result: {record['Correct / Incorrect']} | Category: {record['Confusion Matrix Category']}"
            ),
            paragraph(
                f"Model: {record['OpenAI Model Used']} | Input tokens: {record['Input Tokens']} | "
                f"Output tokens: {record['Output Tokens']} | Status: {record['Evaluation Status']}"
            ),
        ]
    section = '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1080" w:bottom="1440" w:left="1080"/></w:sectPr>'
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body>' + "".join(body) + section + '</w:body></w:document>')


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "reports" / "evaluation_100_openai"
    results = json.loads((output / "evaluation_results.json").read_text(encoding="utf-8"))
    with (output / "prediction_results.csv").open(encoding="utf-8-sig", newline="") as stream:
        prediction_rows = list(csv.DictReader(stream))
    if len(prediction_rows) != 100 or any(row["Evaluation Status"] != "completed" for row in prediction_rows):
        raise SystemExit("The final Word report requires exactly 100 completed prediction records")
    destination = output / "MULTIVENT_100_Review_OpenAI_Evaluation.docx"
    contents = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/></Types>',
        "_rels/.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/></Relationships>',
        "docProps/core.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>MULTIVENT 100-Review OpenAI Evaluation</dc:title><dc:creator>MULTIVENT</dc:creator></cp:coreProperties>',
        "word/document.xml": document_xml(results, prediction_rows),
        "word/_rels/document.xml.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>',
        "word/styles.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:sz w:val="22"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="36"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style></w:styles>',
    }
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in contents.items():
            archive.writestr(name, value.encode("utf-8"))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Generated DOCX archive is corrupt")
    print(destination)


if __name__ == "__main__":
    main()
