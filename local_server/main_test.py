import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from local_server import main
from local_server.tutor import TutorResponse


class GenerateTutorResponseTest(unittest.IsolatedAsyncioTestCase):
    async def test_returns_structured_openai_response(self) -> None:
        expected = TutorResponse(
            correctedText="Minä haluan yhden kahvin.",
            explanation="Use the object form for one complete coffee.",
            reply="Totta kai. Haluatko kahviin maitoa?",
            translation="Of course. Would you like milk in your coffee?",
        )
        parse = AsyncMock(return_value=SimpleNamespace(output_parsed=expected))
        client = SimpleNamespace(responses=SimpleNamespace(parse=parse))

        with (
            patch.object(main, "OPENAI_API_KEY", "test-key"),
            patch.object(main, "AsyncOpenAI", return_value=client),
        ):
            result = await main.generate_tutor_response(
                "Minä haluan yksi kahvi.",
                "Kahvila",
                [],
                "B1",
                "Kirjakieli",
            )

        self.assertEqual(result, expected)
        self.assertEqual(parse.await_args.kwargs["model"], main.OPENAI_MODEL)
        self.assertIs(parse.await_args.kwargs["text_format"], TutorResponse)
        self.assertIn("CEFR B1", parse.await_args.kwargs["instructions"])
        self.assertIn("standard written Finnish", parse.await_args.kwargs["instructions"])

    async def test_rejects_missing_api_key(self) -> None:
        with patch.object(main, "OPENAI_API_KEY", ""):
            with self.assertRaisesRegex(HTTPException, "OPENAI_API_KEY"):
                await main.generate_tutor_response("Hei!", "Arki", [])


class UsageTest(unittest.IsolatedAsyncioTestCase):
    async def test_usage_returns_capped_percentage(self) -> None:
        with (
            patch.object(main, "OPENAI_ADMIN_API_KEY", ""),
            patch.object(main, "estimated_spend_usd", 4.5),
            patch.object(main, "OPENAI_BUDGET_USD", 5.0),
        ):
            result = await main.usage()

        self.assertEqual(result["estimatedSpendUsd"], 4.5)
        self.assertEqual(result["percentage"], 90.0)
        self.assertEqual(result["scope"], "Sovelluksen paikallinen seuranta")

    async def test_usage_returns_openai_platform_costs_with_admin_key(self) -> None:
        amount = SimpleNamespace(value=1.25, currency="usd")
        cost = SimpleNamespace(object="organization.costs.result", amount=amount)
        costs = AsyncMock(
            return_value=SimpleNamespace(data=[SimpleNamespace(results=[cost])])
        )
        client = SimpleNamespace(
            admin=SimpleNamespace(
                organization=SimpleNamespace(usage=SimpleNamespace(costs=costs))
            )
        )
        with (
            patch.object(main, "OPENAI_ADMIN_API_KEY", "admin-key"),
            patch.object(main, "OPENAI_BUDGET_USD", 5.0),
            patch.object(main, "AsyncOpenAI", return_value=client) as openai_client,
        ):
            result = await main.usage()

        self.assertEqual(result["estimatedSpendUsd"], 1.25)
        self.assertEqual(result["percentage"], 25.0)
        self.assertEqual(result["scope"], "OpenAI Platform, tämä kuukausi")
        openai_client.assert_called_once_with(admin_api_key="admin-key", timeout=15)

    def test_estimated_spend_persists_across_restarts(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            usage_path = Path(temporary_directory) / "usage.json"
            with patch.object(main, "USAGE_STATE_PATH", usage_path):
                main.save_estimated_spend(1.25)
                self.assertEqual(main.load_estimated_spend(), 1.25)


if __name__ == "__main__":
    unittest.main()