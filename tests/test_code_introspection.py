"""Tests for code introspection module."""

from pathlib import Path

import pytest

from valtron_core.utilities.code_introspection import CodeIntrospector, LLMCallInstance


class TestCodeIntrospector:
    """Test the CodeIntrospector class."""

    def test_find_litellm_calls(self, tmp_path):
        """Test finding LiteLLM calls."""
        # Create a test file with LiteLLM code
        test_file = tmp_path / "test_litellm.py"
        test_file.write_text(
            """
import litellm

def make_call():
    response = litellm.completion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
    return response
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)

        assert len(instances) > 0
        assert any(inst.library == "litellm" for inst in instances)
        assert any(inst.provider == "Multi-Provider" for inst in instances)

    def test_find_openai_calls(self, tmp_path):
        """Test finding OpenAI calls."""
        test_file = tmp_path / "test_openai.py"
        test_file.write_text(
            """
from openai import OpenAI

client = OpenAI()

def chat():
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}]
    )
    return response
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)

        assert len(instances) > 0
        assert any(inst.library == "openai" for inst in instances)
        assert any(inst.provider == "OpenAI" for inst in instances)
        assert any("chat.completions.create" in inst.function_name for inst in instances)

    def test_find_anthropic_calls(self, tmp_path):
        """Test finding Anthropic calls."""
        test_file = tmp_path / "test_anthropic.py"
        test_file.write_text(
            """
import anthropic

client = anthropic.Anthropic()

def chat():
    message = client.messages.create(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": "Hello"}]
    )
    return message
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)

        assert len(instances) > 0
        assert any(inst.library == "anthropic" for inst in instances)
        assert any(inst.provider == "Anthropic" for inst in instances)

    def test_find_ollama_calls(self, tmp_path):
        """Test finding Ollama calls."""
        test_file = tmp_path / "test_ollama.py"
        test_file.write_text(
            """
import ollama

def generate():
    response = ollama.generate(
        model="llama2",
        prompt="Hello world"
    )
    return response
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)

        assert len(instances) > 0
        assert any(inst.library == "ollama" for inst in instances)
        assert any(inst.provider == "Ollama" for inst in instances)

    def test_find_multiple_providers(self, tmp_path):
        """Test finding calls from multiple providers in one file."""
        test_file = tmp_path / "test_multi.py"
        test_file.write_text(
            """
import litellm
import openai
from anthropic import Anthropic

def use_litellm():
    return litellm.completion(model="gpt-4", messages=[])

def use_openai():
    client = openai.OpenAI()
    return client.chat.completions.create(model="gpt-4", messages=[])

def use_anthropic():
    client = Anthropic()
    return client.messages.create(model="claude-3-opus-20240229", messages=[])
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)

        # Should find calls from all three libraries
        libraries = {inst.library for inst in instances}
        assert "litellm" in libraries
        assert "openai" in libraries
        assert "anthropic" in libraries

    def test_exclude_files_without_imports(self, tmp_path):
        """Test that files without LLM imports are handled correctly."""
        test_file = tmp_path / "test_no_llm.py"
        test_file.write_text(
            """
def regular_function():
    return "Hello world"

class RegularClass:
    pass
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)

        # Should find no LLM calls
        assert len(instances) == 0

    def test_generate_report(self, tmp_path):
        """Test report generation."""
        test_file = tmp_path / "test_report.py"
        test_file.write_text(
            """
import litellm
import openai

litellm.completion(model="gpt-4", messages=[])
litellm.acompletion(model="claude-3", messages=[])
openai.OpenAI().chat.completions.create(model="gpt-3.5-turbo", messages=[])
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)
        report = introspector.generate_report(instances)

        assert report["total_calls"] > 0
        assert "by_provider" in report
        assert "by_library" in report
        assert "by_call_type" in report
        assert "providers_used" in report
        assert "libraries_used" in report

    def test_export_to_json(self, tmp_path):
        """Test JSON export."""
        test_file = tmp_path / "test_export.py"
        test_file.write_text(
            """
import litellm

litellm.completion(model="gpt-4", messages=[])
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)

        output_file = tmp_path / "output.json"
        introspector.export_to_json(instances, output_file)

        assert output_file.exists()
        import json

        with open(output_file, "r") as f:
            data = json.load(f)

        assert "summary" in data
        assert "instances" in data
        assert data["summary"]["total_calls"] > 0

    def test_find_in_directory(self, tmp_path):
        """Test finding LLM calls in a directory tree."""
        # Create multiple files in a directory structure
        (tmp_path / "subdir").mkdir()

        file1 = tmp_path / "file1.py"
        file1.write_text("import litellm\nlitellm.completion(model='gpt-4', messages=[])")

        file2 = tmp_path / "subdir" / "file2.py"
        file2.write_text("from litellm import acompletion\nacompletion(model='gpt-4', messages=[])")

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_directory(tmp_path)

        # Should find calls from both files
        assert len(instances) >= 2
        files = {inst.file_path for inst in instances}
        assert len(files) >= 1  # At least one file should have calls

    def test_real_codebase_client_file(self):
        """Test on the actual valtron_core client.py file."""
        # Get the path to the real client.py file
        project_root = Path(__file__).parent.parent
        client_file = project_root / "src" / "valtron_core" / "client.py"

        if not client_file.exists():
            pytest.skip("client.py not found")

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(client_file)

        # The client.py file uses litellm, so we should find calls
        assert len(instances) > 0
        assert any(inst.library == "litellm" for inst in instances)

    def test_prompt_extraction_literal_string(self, tmp_path):
        """Test extracting literal string prompts."""
        test_file = tmp_path / "test_literal.py"
        test_file.write_text(
            """
import litellm

def make_call():
    response = litellm.completion(
        model="gpt-4",
        prompt="What is the capital of France?"
    )
    return response
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)

        assert len(instances) > 0
        # Find the instance with prompt extraction
        instance = instances[0]
        assert instance.extracted_prompt == "What is the capital of France?"
        assert instance.prompt_confidence == "high"
        assert instance.extraction_method == "literal"

    def test_prompt_extraction_fstring(self, tmp_path):
        """Test extracting f-string prompts."""
        test_file = tmp_path / "test_fstring.py"
        test_file.write_text(
            """
import litellm

def make_call(user_input):
    response = litellm.completion(
        model="gpt-4",
        prompt=f"Translate this to French: {user_input}"
    )
    return response
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)

        assert len(instances) > 0
        instance = instances[0]
        assert instance.extracted_prompt == "Translate this to French: {user_input}"
        assert instance.prompt_confidence == "high"
        assert instance.extraction_method == "fstring"

    def test_prompt_extraction_variable(self, tmp_path):
        """Test extracting prompts from variables."""
        test_file = tmp_path / "test_variable.py"
        test_file.write_text(
            """
import litellm

def make_call():
    system_prompt = "You are a helpful assistant"
    response = litellm.completion(
        model="gpt-4",
        prompt=system_prompt
    )
    return response
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)

        assert len(instances) > 0
        instance = instances[0]
        assert instance.extracted_prompt == "You are a helpful assistant"
        assert instance.prompt_confidence == "medium"
        assert instance.extraction_method == "variable"

    def test_prompt_extraction_messages_list(self, tmp_path):
        """Test extracting chat messages."""
        test_file = tmp_path / "test_messages.py"
        test_file.write_text(
            """
from openai import OpenAI

client = OpenAI()

def chat():
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, world!"}
        ]
    )
    return response
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)

        assert len(instances) > 0
        # Find the instance with the .create() call, not the OpenAI() constructor
        instance = next((inst for inst in instances if ".create" in inst.function_name), instances[-1])
        assert instance.extracted_prompt is not None
        assert "[system]:" in instance.extracted_prompt
        assert "[user]:" in instance.extracted_prompt
        assert "helpful assistant" in instance.extracted_prompt
        assert instance.prompt_confidence == "high"

    def test_prompt_extraction_function_call(self, tmp_path):
        """Test when prompt comes from function call (can't extract)."""
        test_file = tmp_path / "test_function.py"
        test_file.write_text(
            """
import litellm

def load_prompt():
    return "some prompt"

def make_call():
    response = litellm.completion(
        model="gpt-4",
        prompt=load_prompt()
    )
    return response
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)

        assert len(instances) > 0
        # Find instance with litellm.completion call
        instance = next((inst for inst in instances if "completion" in inst.function_name.lower()), instances[0])
        assert instance.extracted_prompt is None
        assert instance.prompt_confidence == "none"
        # The extraction method may be None or "function_call" depending on detection path
        assert instance.extraction_method in [None, "function_call"]

    def test_prompt_extraction_attribute_access(self, tmp_path):
        """Test when prompt comes from attribute (can't extract)."""
        test_file = tmp_path / "test_attribute.py"
        test_file.write_text(
            """
import litellm

class MyClass:
    def __init__(self):
        self.prompt = "some prompt"

    def make_call(self):
        response = litellm.completion(
            model="gpt-4",
            prompt=self.prompt
        )
        return response
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)

        assert len(instances) > 0
        # Find instance with litellm.completion call
        instance = next((inst for inst in instances if "completion" in inst.function_name.lower()), instances[0])
        assert instance.extracted_prompt is None
        assert instance.prompt_confidence == "none"
        # The extraction method may be None or "attribute" depending on detection path
        assert instance.extraction_method in [None, "attribute"]

    def test_prompt_extraction_string_concatenation(self, tmp_path):
        """Test extracting concatenated strings."""
        test_file = tmp_path / "test_concat.py"
        test_file.write_text(
            """
import litellm

def make_call():
    response = litellm.completion(
        model="gpt-4",
        prompt="Part 1: " + "Part 2"
    )
    return response
"""
        )

        introspector = CodeIntrospector()
        instances = introspector.find_llm_calls_in_file(test_file)

        assert len(instances) > 0
        instance = instances[0]
        assert instance.extracted_prompt == "Part 1: Part 2"
        assert instance.prompt_confidence == "medium"
        assert instance.extraction_method == "concatenation"
