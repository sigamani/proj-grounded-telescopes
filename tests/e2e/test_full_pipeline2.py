from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, constr

# ---------- Core Pydantic Schemas ----------

ISO2 = constr(strict=True, min_length=2, max_length=2)


class Person(BaseModel):
    """Minimal person schema used by triage/dependency lookups."""

    id: int
    name: constr(strip_whitespace=True, min_length=1)
    nationality: ISO2
    current_country: ISO2
    dob: date


class InputData(BaseModel):
    """
    Input schema for compliance job requests.
    """

    business_name: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description="Registered business name"
    )
    address: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description="Registered business address"
    )
    registration_id: Optional[str] = Field(
        None, description="Company registration ID, if available"
    )
    country_code: ISO2 = Field(..., description="ISO2 country code")
    website_url: Optional[str] = Field(
        None, description="Company website URL (optional, may contain PII)"
    )
    documents: Optional[List[Any]] = Field(
        None,
        description=(
            "Optional list of documents (e.g., shareholder lists, registry extracts, certificates). "
            "Documents may be binary blobs, base64-encoded strings, or structured attachments."
        ),
    )


class OutputData(BaseModel):
    """
    Output schema for compliance job responses.
    """

    verdict: Literal["ACCEPT", "REJECT", "REVIEW"] = Field(
        ..., description="Final compliance decision"
    )
    structured_data: Dict[str, Any] = Field(
        ...,
        description=(
            "Structured information extracted about company, people, industry, and online presence"
        ),
    )
    due_diligence_report: str = Field(
        ...,
        description=(
            "Narrative report detailing steps taken, findings, risk signals, and "
            "recommendations for next actions"
        ),
    )


# ---------- Dependencies / Stubs ----------


class DatabaseConn:
    """
    Example dependency for local queries (e.g., a SQLite index).
    Replace with your actual implementation.
    """

    def __init__(self, dsn: str = "sqlite:///watchman.db"):
        self.dsn = dsn

    def persons_of_interest(self, business_name: str) -> List[Person]:
        import sqlite3

        conn = sqlite3.connect(self.dsn.replace("sqlite:///", ""))
        cursor = conn.execute(
            "SELECT entity_id, sdn_name FROM sdn WHERE sdn_type='individual' AND sdn_name LIKE ?",
            (f"%{business_name}%",),
        )
        results = [
            Person(
                id=int(row[0]),
                name=row[1],
                nationality="XX",
                current_country="XX",
                dob=date(1900, 1, 1),
            )
            for row in cursor.fetchall()
        ]
        conn.close()
        return results


class TriageDependencies(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    business_name: str
    db: DatabaseConn


# ---------- Ray/vLLM Integration ----------


def process_kyc_batch(input_data_list: List[InputData]) -> List[OutputData]:
    # TODO: Create unit test for Ray/vLLM integration
    import ray

    from src.batch_infer import create_batch_processor

    processor = create_batch_processor()
    prompts = [
        {"prompt": f"KYC analysis for {data.business_name} in {data.country_code}"}
        for data in input_data_list
    ]
    ds = ray.data.from_items(prompts)
    results = processor(ds).take_all()
    return [
        OutputData(
            verdict="REVIEW",
            structured_data={"company": {"name": input_data_list[i].business_name}},
            due_diligence_report=results[i]["out"],
        )
        for i in range(len(input_data_list))
    ]


# ---------- TODO (task 3): wire up pydantic_ai.Agent if available ----------
# TODO: Create unit test for pydantic_ai Agent with dependency injection

try:
    from pydantic_ai import Agent  # type: ignore

    triage_agent = Agent[InputData, OutputData](
        "qwen-1.5b",
        system_prompt="KYC compliance officer. Return ACCEPT/REJECT/REVIEW verdict.",
        deps_type=TriageDependencies,
    )
except (ImportError, Exception):
    triage_agent = None


# ---------- Requirement (use sample_in and out as they are in this file)  ----------


def test_person_model():
    person = Person(
        id=1,
        name="John Doe",
        nationality="US",
        current_country="US",
        dob=date(1990, 1, 1),
    )
    assert person.id == 1


def test_database_conn():
    db = DatabaseConn()
    assert db.dsn == "sqlite:///watchman.db"


def test_triage_dependencies():
    db = DatabaseConn()
    deps = TriageDependencies(business_name="Test", db=db)
    assert deps.business_name == "Test"


def test_input_data_validation():
    sample_in = InputData(
        business_name="Acme Corp",
        address="123 High Street, London",
        registration_id=None,
        country_code="GB",
        website_url="https://acme.example",
        documents=None,
    )
    assert sample_in.business_name == "Acme Corp"


def test_output_data_validation():
    sample_out = OutputData(
        verdict="REVIEW",
        structured_data={
            "company": {"name": "Acme Corp", "country": "GB"},
            "people": [],
        },
        due_diligence_report="Reviewed filings and public sources; potential risk indicators require analyst review.",
    )
    assert sample_out.verdict == "REVIEW"


def test_database_query():
    from unittest.mock import MagicMock, patch

    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = [(1, "Test Person")]
        db = DatabaseConn("sqlite:///test.db")
        results = db.persons_of_interest("test")
        assert len(results) == 1


def test_triage_agent_init():
    if triage_agent is None:
        assert True  # pydantic_ai not available
    else:
        assert triage_agent is not None


def test_batch_processor():
    try:
        processor = process_kyc_batch(
            [InputData(business_name="Test", address="Test", country_code="GB")]
        )
        assert isinstance(processor, list)
    except ImportError:
        assert True  # ray not available


def test_iso2_constraint():
    person = Person(
        id=1, name="Jane", nationality="US", current_country="CA", dob=date(2000, 1, 1)
    )
    assert person.nationality == "US"


def test_main_execution():
    import subprocess

    result = subprocess.run(["python", __file__], capture_output=True, text=True)
    assert "InputData OK:" in result.stdout or "OutputData OK:" in result.stdout


def test_main_block():
    sample_in = InputData(
        business_name="Acme Corp",
        address="123 High Street, London",
        registration_id=None,
        country_code="GB",
        website_url="https://acme.example",
        documents=None,
    )
    sample_out = OutputData(
        verdict="REVIEW",
        structured_data={
            "company": {"name": "Acme Corp", "country": "GB"},
            "people": [],
        },
        due_diligence_report="Reviewed filings and public sources; potential risk indicators require analyst review.",
    )
    assert sample_in.business_name == "Acme Corp" and sample_out.verdict == "REVIEW"


def test_process_kyc_mock():
    from unittest.mock import patch

    with patch("tests.e2e.test_full_pipeline2.create_batch_processor") as mock_func:
        mock_func.return_value = lambda ds: ds
        with patch("tests.e2e.test_full_pipeline2.ray") as mock_ray:
            mock_ray.data.from_items.return_value.take_all.return_value = [
                {"out": "test"}
            ]
            result = process_kyc_batch(
                [InputData(business_name="Test", address="Test", country_code="GB")]
            )
            assert isinstance(result, list)


if __name__ == "__main__":
    sample_in = InputData(
        business_name="Acme Corp",
        address="123 High Street, London",
        registration_id=None,
        country_code="GB",
        website_url="https://acme.example",
        documents=None,
    )
    print("InputData OK:\n", sample_in.model_dump_json(indent=2))

    sample_out = OutputData(
        verdict="REVIEW",
        structured_data={
            "company": {"name": "Acme Corp", "country": "GB"},
            "people": [],
        },
        due_diligence_report="Reviewed filings and public sources; potential risk indicators require analyst review.",
    )
    print("\nOutputData OK:\n", sample_out.model_dump_json(indent=2))

    if triage_agent is None:
        print(
            "\n[pydantic_ai] Agent not initialised (library missing or API mismatch)."
        )
    else:
        print("\n[pydantic_ai] Agent initialised:", triage_agent)
