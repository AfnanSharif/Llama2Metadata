import tempfile
import unittest
import json
from pathlib import Path

from metadata_rag.extractors import extract_asset
from metadata_rag.index import HashVectorIndex
from metadata_rag.models import Asset
from metadata_rag.providers import HeuristicProvider, Llama2Provider, _validated_metadata
from metadata_rag.service import MetadataStudio


class MetadataTests(unittest.TestCase):
    def test_extraction_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payment_service.py"
            path.write_text("class PaymentService:\n    \"\"\"Processes secure invoices for Nova Corp.\"\"\"\n", encoding="utf-8")
            asset = extract_asset(path)
            metadata = HeuristicProvider().generate(asset)
            self.assertEqual(metadata.content_type, "source-code")
            self.assertTrue(metadata.keywords)
            self.assertTrue(asset.checksum)

    def test_rag_relates_similar_assets(self) -> None:
        first = Asset("one", "api.md", "Python API authentication tokens", "text/markdown", 10, "a")
        second = Asset("two", "guide.md", "API authentication guide for Python", "text/markdown", 10, "b")
        studio = MetadataStudio(index=HashVectorIndex())
        studio.add([first, second])
        result = studio.generate(first)
        self.assertIn("two", result.related_assets)

    def test_generated_metadata_validation_rejects_bad_confidence(self) -> None:
        asset = Asset("one", "guide.md", "Guide text", "text/markdown", 10, "a")
        values = {"title": "Guide", "summary": "Summary", "keywords": ["guide"], "content_type": "document", "language": "en", "entities": [], "confidence": 2}
        with self.assertRaises(ValueError):
            _validated_metadata(asset, values, "test")

    def test_hash_index_validates_dimensions_and_limit(self) -> None:
        with self.assertRaises(ValueError):
            HashVectorIndex(0)
        with self.assertRaises(ValueError):
            HashVectorIndex().search("query", limit=0)

    def test_tika_and_textract_adapters_are_reachable_and_bounded(self) -> None:
        class FakeTika:
            @staticmethod
            def from_file(path):
                return {"content": f"Tika extracted {Path(path).name}"}

        class FakeTextract:
            def __init__(self):
                self.calls = []

            def detect_document_text(self, **kwargs):
                self.calls.append(kwargs)
                return {"Blocks": [{"BlockType": "LINE", "Text": "AWS extracted line"}]}

        with tempfile.TemporaryDirectory() as directory:
            tika_path = Path(directory) / "archive.bin"
            tika_path.write_bytes(b"opaque")
            tika_asset = extract_asset(tika_path, extractor="tika", tika_parser=FakeTika())
            image_path = Path(directory) / "scan.png"
            image_path.write_bytes(b"fake-image")
            textract = FakeTextract()
            textract_asset = extract_asset(image_path, extractor="textract", textract_client=textract)
            with self.assertRaises(ValueError):
                extract_asset(image_path, extractor="textract", textract_client=textract, max_bytes=2)
        self.assertIn("Tika extracted", tika_asset.text)
        self.assertEqual(tika_asset.extractor, "tika")
        self.assertEqual(textract_asset.text, "AWS extracted line")
        self.assertEqual(len(textract.calls), 1)

    def test_service_routes_extraction_adapter(self) -> None:
        calls = []

        def extractor(path, **kwargs):
            calls.append((Path(path).name, kwargs["extractor"]))
            return Asset("asset", Path(path).name, "searchable evidence text", "text/plain", 1, "hash", extractor=kwargs["extractor"])

        studio = MetadataStudio(index=HashVectorIndex(), asset_extractor=extractor)
        assets = studio.ingest(["document.pdf"], extractor="textract")
        self.assertEqual(calls, [("document.pdf", "textract")])
        self.assertEqual(assets[0].extractor, "textract")

    def test_quantized_llama_and_grounded_qa_use_provider_pipeline(self) -> None:
        captured = []
        outputs = [
            {
                "title": "Guide",
                "summary": "A grounded guide.",
                "keywords": ["guide"],
                "content_type": "document",
                "language": "en",
                "entities": [],
                "confidence": 0.8,
            },
            {"answer": "The guide requires signed metadata.", "citations": ["one"]},
        ]

        class FakePipeline:
            def __call__(self, prompt, **kwargs):
                captured.append((prompt, kwargs))
                return [{"generated_text": json.dumps(outputs.pop(0))}]

        def loader(config):
            captured.append(config)
            return FakePipeline()

        provider = Llama2Provider("test/llama", quantization="4bit", device="cuda", pipeline_loader=loader)
        asset = Asset("one", "guide.md", "Signed metadata is required by the guide.", "text/markdown", 20, "x")
        self.assertEqual(provider.generate(asset).generator, "llama2")
        answer = provider.answer("What is required?", [asset])
        self.assertTrue(answer.grounded)
        self.assertEqual(answer.source_ids, ["one"])
        self.assertEqual(captured[0].quantization, "4bit")
        self.assertIn("[SOURCE one", captured[2][0])

    def test_studio_qa_invokes_selected_provider(self) -> None:
        class Provider(HeuristicProvider):
            def __init__(self):
                self.called = False

            def answer(self, question, evidence):
                self.called = True
                return super().answer(question, evidence)

        provider = Provider()
        studio = MetadataStudio(provider, HashVectorIndex())
        studio.add([Asset("one", "guide.md", "Encryption protects stored metadata.", "text/markdown", 10, "x")])
        result = studio.ask("How is metadata protected?")
        self.assertTrue(provider.called)
        self.assertTrue(result.grounded)

    def test_aws_assets_include_ec2_iam_textract_and_bootstrap(self) -> None:
        root = Path(__file__).parents[1]
        policy = json.loads((root / "deploy" / "aws" / "iam-policy.json").read_text(encoding="utf-8"))
        template = (root / "deploy" / "aws" / "ec2-cloudformation.yaml").read_text(encoding="utf-8")
        bootstrap = (root / "deploy" / "aws" / "bootstrap.sh").read_text(encoding="utf-8")
        actions = policy["Statement"][0]["Action"]
        self.assertIn("textract:DetectDocumentText", actions)
        self.assertIn("AWS::EC2::Instance", template)
        self.assertIn("AWS::IAM::InstanceProfile", template)
        self.assertIn("bootstrap.sh", template)
        self.assertIn("streamlit run app.py", bootstrap)


if __name__ == "__main__":
    unittest.main()
