"""Code introspection to find LLM API calls in codebases."""

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

import structlog

logger = structlog.get_logger()


@dataclass
class LLMCallPattern:
    """Represents a pattern for detecting LLM API calls."""

    provider: str
    library: str
    function_pattern: str  # Regex pattern for function/method names
    import_pattern: str  # Pattern for import statements
    call_type: str  # Type of call: "completion", "chat", "embedding", etc.
    examples: List[str]  # Example code snippets


@dataclass
class LLMCallInstance:
    """Represents a found instance of an LLM API call."""

    file_path: str
    line_number: int
    provider: str
    library: str
    call_type: str
    code_snippet: str
    function_name: str
    confidence: float  # 0-1 confidence score

    # Prompt extraction fields
    extracted_prompt: str | None = None  # The extracted prompt text (if found)
    prompt_confidence: str = "none"  # "high", "medium", "low", "none"
    extraction_method: str | None = None  # How prompt was extracted: "literal", "variable", "fstring", etc.
    extraction_notes: str | None = None  # Why prompt couldn't be extracted or additional info


# Comprehensive list of LLM API call patterns
LLM_CALL_PATTERNS: List[LLMCallPattern] = [
    # ==================== LiteLLM ====================
    LLMCallPattern(
        provider="Multi-Provider",
        library="litellm",
        function_pattern=r"litellm\.(completion|acompletion|embedding|aembedding|image_generation|aimage_generation|completion_cost|model_cost|get_model_info)",
        import_pattern=r"from litellm import|import litellm",
        call_type="completion",
        examples=[
            "litellm.completion(model='gpt-4', messages=messages)",
            "await litellm.acompletion(model='claude-3', messages=messages)",
            "litellm.embedding(model='text-embedding-ada-002', input=text)",
        ],
    ),
    # ==================== OpenAI ====================
    LLMCallPattern(
        provider="OpenAI",
        library="openai",
        function_pattern=r"(openai\.|client\.)chat\.completions\.create|ChatCompletion\.create|Completion\.create",
        import_pattern=r"from openai import|import openai",
        call_type="chat_completion",
        examples=[
            "openai.chat.completions.create(model='gpt-4', messages=messages)",
            "client.chat.completions.create(model='gpt-3.5-turbo', messages=messages)",
            "openai.ChatCompletion.create(model='gpt-4', messages=messages)",  # Old API
        ],
    ),
    LLMCallPattern(
        provider="OpenAI",
        library="openai",
        function_pattern=r"(openai\.|client\.)completions\.create",
        import_pattern=r"from openai import|import openai",
        call_type="completion",
        examples=[
            "openai.completions.create(model='text-davinci-003', prompt=prompt)",
            "client.completions.create(model='gpt-3.5-turbo-instruct', prompt=prompt)",
        ],
    ),
    LLMCallPattern(
        provider="OpenAI",
        library="openai",
        function_pattern=r"(openai\.|client\.)embeddings\.create",
        import_pattern=r"from openai import|import openai",
        call_type="embedding",
        examples=[
            "openai.embeddings.create(model='text-embedding-ada-002', input=text)",
            "client.embeddings.create(model='text-embedding-3-small', input=text)",
        ],
    ),
    LLMCallPattern(
        provider="OpenAI",
        library="openai",
        function_pattern=r"(openai\.|client\.)images\.generate",
        import_pattern=r"from openai import|import openai",
        call_type="image_generation",
        examples=[
            "openai.images.generate(model='dall-e-3', prompt=prompt)",
            "client.images.generate(model='dall-e-2', prompt=prompt)",
        ],
    ),
    # ==================== Anthropic ====================
    LLMCallPattern(
        provider="Anthropic",
        library="anthropic",
        function_pattern=r"(anthropic\.|client\.)messages\.create|completions\.create",
        import_pattern=r"from anthropic import|import anthropic",
        call_type="completion",
        examples=[
            "anthropic.messages.create(model='claude-3-opus-20240229', messages=messages)",
            "client.messages.create(model='claude-3-sonnet-20240229', messages=messages)",
            "anthropic.completions.create(model='claude-2', prompt=prompt)",  # Old API
        ],
    ),
    LLMCallPattern(
        provider="Anthropic",
        library="anthropic",
        function_pattern=r"(anthropic\.|client\.)messages\.stream",
        import_pattern=r"from anthropic import|import anthropic",
        call_type="streaming",
        examples=[
            "anthropic.messages.stream(model='claude-3-opus-20240229', messages=messages)",
            "client.messages.stream(model='claude-3-sonnet-20240229', messages=messages)",
        ],
    ),
    # ==================== Google (Gemini/PaLM) ====================
    LLMCallPattern(
        provider="Google",
        library="google.generativeai",
        function_pattern=r"genai\.generate_text|client\.models\.generate_content|model\.generate_content|model\.generate_content_async|models\.generate_content",
        import_pattern=r"import google\.generativeai|from google\.generativeai import|from google import genai",
        call_type="completion",
        examples=[
            "model.generate_content('What is AI?')",
            "genai.generate_text(prompt='Hello', model='models/text-bison-001')",
            "client.models.generate_content(model='gemini-2.5-pro', contents=prompt)",
        ],
    ),
    LLMCallPattern(
        provider="Google",
        library="vertexai",
        function_pattern=r"model\.predict|model\.predict_async",
        import_pattern=r"from vertexai\.language_models import|from vertexai\.preview\.language_models import",
        call_type="completion",
        examples=[
            "model.predict('What is AI?')",
            "model.predict_async('What is AI?')",
        ],
    ),
    # ==================== Cohere ====================
    LLMCallPattern(
        provider="Cohere",
        library="cohere",
        function_pattern=r"(cohere\.|client\.|co\.)generate|chat|embed|classify|summarize|rerank",
        import_pattern=r"import cohere|from cohere import",
        call_type="completion",
        examples=[
            "co.generate(model='command', prompt='Hello')",
            "client.chat(model='command-r', message='Hello')",
            "co.embed(model='embed-english-v3.0', texts=texts)",
        ],
    ),
    # ==================== Hugging Face ====================
    LLMCallPattern(
        provider="HuggingFace",
        library="transformers",
        function_pattern=r"pipeline\(|AutoModelForCausalLM|AutoModelForSeq2SeqLM|model\.generate|model\(\)",
        import_pattern=r"from transformers import|import transformers",
        call_type="completion",
        examples=[
            "pipeline('text-generation', model='gpt2')",
            "model = AutoModelForCausalLM.from_pretrained('gpt2')",
            "model.generate(input_ids)",
        ],
    ),
    LLMCallPattern(
        provider="HuggingFace",
        library="huggingface_hub",
        function_pattern=r"InferenceClient|InferenceApi|hf_hub_download|text_generation|chat_completion",
        import_pattern=r"from huggingface_hub import|import huggingface_hub",
        call_type="completion",
        examples=[
            "client = InferenceClient()",
            "client.text_generation('Hello', model='gpt2')",
            "client.chat_completion(messages, model='mistralai/Mistral-7B')",
        ],
    ),
    # ==================== LangChain ====================
    LLMCallPattern(
        provider="Multi-Provider",
        library="langchain",
        function_pattern=r"(OpenAI|ChatOpenAI|Anthropic|ChatAnthropic|GooglePalm|Cohere|HuggingFaceHub|Ollama)\(|llm\.predict|llm\.invoke|llm\.ainvoke|chain\.run|chain\.invoke",
        import_pattern=r"from langchain\.|from langchain_|import langchain",
        call_type="completion",
        examples=[
            "llm = ChatOpenAI(model='gpt-4')",
            "llm = ChatAnthropic(model='claude-3-opus-20240229')",
            "llm.invoke('What is AI?')",
            "chain.run(input='Hello')",
        ],
    ),
    # ==================== LlamaIndex ====================
    LLMCallPattern(
        provider="Multi-Provider",
        library="llama_index",
        function_pattern=r"OpenAI\(|ChatOpenAI\(|Anthropic\(|PaLM\(|llm\.complete|llm\.chat|llm\.stream_complete",
        import_pattern=r"from llama_index\.llms import|import llama_index",
        call_type="completion",
        examples=[
            "llm = OpenAI(model='gpt-4')",
            "llm.complete('What is AI?')",
            "llm.chat(messages)",
        ],
    ),
    # ==================== Replicate ====================
    LLMCallPattern(
        provider="Replicate",
        library="replicate",
        function_pattern=r"replicate\.run|client\.run|replicate\.models\.get|model\.predict",
        import_pattern=r"import replicate|from replicate import",
        call_type="completion",
        examples=[
            "replicate.run('stability-ai/sdxl', input={'prompt': 'a cat'})",
            "client.run('meta/llama-2-70b', input={'prompt': 'Hello'})",
        ],
    ),
    # ==================== Together AI ====================
    LLMCallPattern(
        provider="Together",
        library="together",
        function_pattern=r"together\.(Complete|Chat|Embedding|Image)\.create|client\.(completions|chat|embeddings)\.create",
        import_pattern=r"import together|from together import",
        call_type="completion",
        examples=[
            "together.Complete.create(prompt='Hello', model='togethercomputer/llama-2-70b')",
            "together.Chat.create(messages=messages, model='mistralai/Mixtral-8x7B')",
        ],
    ),
    # ==================== Ollama ====================
    LLMCallPattern(
        provider="Ollama",
        library="ollama",
        function_pattern=r"ollama\.(generate|chat|embeddings|create|list|show)|client\.(generate|chat|embeddings)",
        import_pattern=r"import ollama|from ollama import",
        call_type="completion",
        examples=[
            "ollama.generate(model='llama2', prompt='Hello')",
            "ollama.chat(model='llama2', messages=messages)",
            "client.generate(model='mistral', prompt='What is AI?')",
        ],
    ),
    # ==================== Azure OpenAI ====================
    LLMCallPattern(
        provider="Azure OpenAI",
        library="openai",
        function_pattern=r"AzureOpenAI\(|azure_endpoint=",
        import_pattern=r"from openai import AzureOpenAI",
        call_type="completion",
        examples=[
            "client = AzureOpenAI(azure_endpoint=endpoint, api_key=key)",
            "client.chat.completions.create(model='gpt-4', messages=messages)",
        ],
    ),
    # ==================== AWS Bedrock ====================
    LLMCallPattern(
        provider="AWS Bedrock",
        library="boto3",
        function_pattern=r"client\('bedrock-runtime'\)|invoke_model|invoke_model_with_response_stream",
        import_pattern=r"import boto3",
        call_type="completion",
        examples=[
            "client = boto3.client('bedrock-runtime')",
            "client.invoke_model(modelId='anthropic.claude-v2', body=body)",
        ],
    ),
    # ==================== Mistral AI ====================
    LLMCallPattern(
        provider="Mistral",
        library="mistralai",
        function_pattern=r"MistralClient|client\.chat|client\.embeddings|client\.chat_stream",
        import_pattern=r"from mistralai\.client import|import mistralai",
        call_type="completion",
        examples=[
            "client = MistralClient(api_key=api_key)",
            "client.chat(model='mistral-medium', messages=messages)",
        ],
    ),
    # ==================== AI21 ====================
    LLMCallPattern(
        provider="AI21",
        library="ai21",
        function_pattern=r"ai21\.(Completion|Chat|Summarize)\.execute|client\.(complete|chat)",
        import_pattern=r"import ai21|from ai21 import",
        call_type="completion",
        examples=[
            "ai21.Completion.execute(prompt='Hello', model='j2-ultra')",
            "client.complete(prompt='What is AI?', model='j2-mid')",
        ],
    ),
    # ==================== Anyscale ====================
    LLMCallPattern(
        provider="Anyscale",
        library="openai",
        function_pattern=r"base_url.*anyscale|api\.endpoints\.anyscale",
        import_pattern=r"from openai import",
        call_type="completion",
        examples=[
            "client = OpenAI(base_url='https://api.endpoints.anyscale.com/v1')",
        ],
    ),
    # ==================== Perplexity ====================
    LLMCallPattern(
        provider="Perplexity",
        library="openai",
        function_pattern=r"base_url.*perplexity|api\.perplexity",
        import_pattern=r"from openai import",
        call_type="completion",
        examples=[
            "client = OpenAI(base_url='https://api.perplexity.ai')",
        ],
    ),
]


class CodeIntrospector:
    """Introspects code to find LLM API calls."""

    def __init__(self, patterns: List[LLMCallPattern] | None = None):
        """
        Initialize the code introspector.

        Args:
            patterns: Optional list of LLM call patterns to search for
        """
        self.patterns = patterns or LLM_CALL_PATTERNS
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for faster matching."""
        self.compiled_patterns = []
        for pattern in self.patterns:
            try:
                compiled_func = re.compile(pattern.function_pattern)
                compiled_import = re.compile(pattern.import_pattern)
                self.compiled_patterns.append((pattern, compiled_func, compiled_import))
            except re.error as e:
                logger.warning("invalid_pattern", pattern=pattern.function_pattern, error=str(e))

    def find_llm_calls_in_file(self, file_path: Path) -> List[LLMCallInstance]:
        """
        Find all LLM API calls in a single file and extract prompts.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            List of found LLM call instances with extracted prompts
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

            instances = []

            # First, detect which libraries are imported
            imported_libraries = self._detect_imports(content)
            imported_names = self._extract_imported_names(content)

            # Parse AST once and build a map of line numbers to Call nodes for prompt extraction
            line_to_call_nodes = self._build_line_to_call_map(content)

            # Search for function calls line by line
            for line_num, line in enumerate(lines, start=1):
                for pattern, func_regex, import_regex in self.compiled_patterns:
                    # Only check patterns for libraries that are imported
                    if pattern.library not in imported_libraries:
                        continue

                    match = func_regex.search(line)
                    if match:
                        # Extract prompt for this line
                        extracted_prompt, prompt_conf, extraction_method, notes = self._extract_prompt_for_line(
                            line_num, line_to_call_nodes, content
                        )

                        instances.append(
                            LLMCallInstance(
                                file_path=str(file_path),
                                line_number=line_num,
                                provider=pattern.provider,
                                library=pattern.library,
                                call_type=pattern.call_type,
                                code_snippet=line.strip(),
                                function_name=match.group(0),
                                confidence=0.9,  # High confidence for direct matches
                                extracted_prompt=extracted_prompt,
                                prompt_confidence=prompt_conf,
                                extraction_method=extraction_method,
                                extraction_notes=notes,
                            )
                        )

                # Also check for imported function calls (e.g., "completion()" when "from litellm import completion")
                for lib_name, func_names in imported_names.items():
                    for func_name in func_names:
                        # Match function calls like: func_name(...)
                        func_call_pattern = rf'\b{re.escape(func_name)}\s*\('
                        if re.search(func_call_pattern, line):
                            # Find matching pattern for this library
                            for pattern, _, _ in self.compiled_patterns:
                                if pattern.library == lib_name or lib_name.startswith(pattern.library):
                                    # Extract prompt for this line
                                    extracted_prompt, prompt_conf, extraction_method, notes = self._extract_prompt_for_line(
                                        line_num, line_to_call_nodes, content
                                    )

                                    instances.append(
                                        LLMCallInstance(
                                            file_path=str(file_path),
                                            line_number=line_num,
                                            provider=pattern.provider,
                                            library=pattern.library,
                                            call_type=pattern.call_type,
                                            code_snippet=line.strip(),
                                            function_name=func_name,
                                            confidence=0.85,  # Slightly lower confidence
                                            extracted_prompt=extracted_prompt,
                                            prompt_confidence=prompt_conf,
                                            extraction_method=extraction_method,
                                            extraction_notes=notes,
                                        )
                                    )
                                    break  # Only add once per pattern match

            # Also try AST-based detection for more complex cases (already extracts prompts)
            ast_instances = self._find_with_ast(file_path, content, imported_libraries)
            instances.extend(ast_instances)

            # Remove duplicates
            unique_instances = self._deduplicate_instances(instances)

            logger.info(
                "file_analyzed",
                file=str(file_path),
                instances_found=len(unique_instances),
            )

            return unique_instances

        except Exception as e:
            logger.error("file_analysis_failed", file=str(file_path), error=str(e))
            return []

    def _detect_imports(self, content: str) -> Set[str]:
        """
        Detect which LLM libraries are imported.

        Args:
            content: File content

        Returns:
            Set of library names that are imported
        """
        imported = set()
        for pattern, _, import_regex in self.compiled_patterns:
            if import_regex.search(content):
                imported.add(pattern.library)
        return imported

    def _extract_imported_names(self, content: str) -> Dict[str, Set[str]]:
        """
        Extract which specific functions are imported from each library.

        Args:
            content: File content

        Returns:
            Dict mapping library name to set of imported function names
        """
        imported_names: Dict[str, Set[str]] = {}

        # Parse import statements
        import_patterns = [
            # from module import func1, func2
            (r'from\s+(\S+)\s+import\s+([^#\n]+)', 'from_import'),
            # import module
            (r'import\s+(\S+)(?:\s+as\s+\S+)?', 'direct_import'),
        ]

        for line in content.split('\n'):
            line = line.strip()
            if not line.startswith(('import ', 'from ')):
                continue

            # from X import Y pattern
            from_match = re.match(r'from\s+(\S+)\s+import\s+([^#\n]+)', line)
            if from_match:
                module = from_match.group(1)
                imports = from_match.group(2)

                # Parse individual imports (handle commas, parentheses)
                imports = imports.replace('(', '').replace(')', '')
                func_names = [name.strip().split(' as ')[0].strip()
                             for name in imports.split(',')]

                if module not in imported_names:
                    imported_names[module] = set()
                imported_names[module].update(func_names)

        return imported_names

    def _build_line_to_call_map(self, content: str) -> Dict[int, List[ast.Call]]:
        """
        Build a map of line numbers to AST Call nodes.

        Args:
            content: File content

        Returns:
            Dict mapping line numbers to list of Call nodes on that line
        """
        line_to_calls: Dict[int, List[ast.Call]] = {}

        try:
            tree = ast.parse(content)

            class CallCollector(ast.NodeVisitor):
                def __init__(self):
                    self.tree = tree  # Store tree for later use

                def visit_Call(self, node):
                    if node.lineno not in line_to_calls:
                        line_to_calls[node.lineno] = []
                    line_to_calls[node.lineno].append(node)
                    self.generic_visit(node)

            collector = CallCollector()
            collector.visit(tree)

        except SyntaxError:
            pass  # If AST parsing fails, return empty map

        return line_to_calls

    def _extract_prompt_for_line(
        self, line_num: int, line_to_call_nodes: Dict[int, List[ast.Call]], content: str
    ) -> tuple[str | None, str, str | None, str | None]:
        """
        Extract prompt for a specific line number.

        Args:
            line_num: Line number where the call is located
            line_to_call_nodes: Map of line numbers to Call nodes
            content: Full file content

        Returns:
            Tuple of (extracted_prompt, confidence, method, notes)
        """
        # If we don't have Call nodes for this line, we can't extract
        if line_num not in line_to_call_nodes:
            return None, "none", None, "Could not parse AST for this line"

        # Try each Call node on this line (usually just one)
        # Use the first one that successfully extracts a prompt
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None, "none", None, "AST parsing failed"

        for call_node in line_to_call_nodes[line_num]:
            extracted, conf, method, notes = self._extract_prompt_from_ast(call_node, content, tree)
            if extracted or conf != "none":
                return extracted, conf, method, notes

        # If we get here, we found Call nodes but couldn't extract prompts
        return None, "none", None, "No prompt argument found in function call"

    def _find_with_ast(
        self, file_path: Path, content: str, imported_libraries: Set[str]
    ) -> List[LLMCallInstance]:
        """
        Use AST parsing to find LLM calls and extract prompts.

        Args:
            file_path: Path to file
            content: File content
            imported_libraries: Set of imported library names

        Returns:
            List of found instances with extracted prompts
        """
        instances = []

        try:
            tree = ast.parse(content)

            class CallVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.calls = []  # Store (line_num, func_name, node)

                def visit_Call(self, node):
                    # Extract function name from various call patterns
                    func_name = self._get_full_name(node.func)
                    if func_name:
                        self.calls.append((node.lineno, func_name, node))
                    self.generic_visit(node)

                def _get_full_name(self, node):
                    """Get full dotted name from AST node."""
                    if isinstance(node, ast.Name):
                        return node.id
                    elif isinstance(node, ast.Attribute):
                        value = self._get_full_name(node.value)
                        if value:
                            return f"{value}.{node.attr}"
                        return node.attr
                    return None

            visitor = CallVisitor()
            visitor.visit(tree)

            # Match found calls against patterns
            for line_num, func_name, call_node in visitor.calls:
                for pattern, func_regex, _ in self.compiled_patterns:
                    if pattern.library not in imported_libraries:
                        continue

                    if func_regex.search(func_name):
                        # Get the actual line of code
                        lines = content.split("\n")
                        code_snippet = lines[line_num - 1].strip() if line_num <= len(lines) else ""

                        # Extract prompt from the call
                        extracted_prompt, prompt_conf, extraction_method, notes = self._extract_prompt_from_ast(
                            call_node, content, tree
                        )

                        instances.append(
                            LLMCallInstance(
                                file_path=str(file_path),
                                line_number=line_num,
                                provider=pattern.provider,
                                library=pattern.library,
                                call_type=pattern.call_type,
                                code_snippet=code_snippet,
                                function_name=func_name,
                                confidence=0.85,  # Slightly lower confidence for AST matches
                                extracted_prompt=extracted_prompt,
                                prompt_confidence=prompt_conf,
                                extraction_method=extraction_method,
                                extraction_notes=notes,
                            )
                        )

        except SyntaxError as e:
            logger.warning("ast_parse_failed", file=str(file_path), error=str(e))

        return instances

    def _deduplicate_instances(self, instances: List[LLMCallInstance]) -> List[LLMCallInstance]:
        """
        Remove duplicate instances (same file, line, function).

        Args:
            instances: List of instances

        Returns:
            Deduplicated list
        """
        seen = set()
        unique = []

        for instance in instances:
            key = (instance.file_path, instance.line_number, instance.function_name)
            if key not in seen:
                seen.add(key)
                unique.append(instance)

        return unique

    def _extract_prompt_from_ast(
        self, node: ast.Call, content: str, tree: ast.AST
    ) -> tuple[str | None, str, str | None, str | None]:
        """
        Extract prompt from an AST Call node.

        Args:
            node: The Call node representing the LLM API call
            content: Full file content for context
            tree: Full AST tree for variable lookups

        Returns:
            Tuple of (extracted_prompt, confidence, method, notes)
            - extracted_prompt: The extracted prompt string or None
            - confidence: "high", "medium", "low", or "none"
            - method: How it was extracted (e.g., "literal", "variable", "fstring")
            - notes: Additional information or reason for failure
        """
        # Common prompt argument names across different LLM APIs
        prompt_arg_names = [
            "prompt",
            "messages",
            "contents",
            "content",
            "input",
            "text",
            "query",
            "system",
        ]

        # Try to find prompt in keyword arguments
        for keyword in node.keywords:
            if keyword.arg in prompt_arg_names:
                return self._extract_value_from_node(keyword.value, content, tree, node.lineno)

        # Try to find in positional arguments (less common but possible)
        if node.args:
            # For some APIs, the prompt is the first positional arg
            return self._extract_value_from_node(node.args[0], content, tree, node.lineno)

        return None, "none", None, "No prompt argument found in function call"

    def _extract_value_from_node(
        self, node: ast.expr, content: str, tree: ast.AST, call_line: int
    ) -> tuple[str | None, str, str | None, str | None]:
        """
        Extract string value from an AST expression node.

        Args:
            node: AST expression node
            content: Full file content
            tree: Full AST tree for variable lookups
            call_line: Line number of the call

        Returns:
            Tuple of (value, confidence, method, notes)
        """
        # Case 1: Literal string (highest confidence)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value, "high", "literal", "Direct string literal"

        # Case 2: F-string / JoinedStr (high confidence)
        if isinstance(node, ast.JoinedStr):
            try:
                result = self._reconstruct_fstring(node)
                return result, "high", "fstring", "F-string with template variables"
            except Exception as e:
                return None, "low", "fstring", f"Could not fully reconstruct f-string: {str(e)}"

        # Case 3: List (could be message dictionaries OR mixed content like [prompt, file])
        if isinstance(node, ast.List):
            # First try as message dictionaries (OpenAI, Anthropic style)
            messages = self._extract_messages_list(node, content, tree, call_line)
            if messages:
                return messages, "high", "message_list", "Chat messages array"

            # If not message dicts, try extracting string elements (Google Gemini style)
            # Look for variables or string literals in the list
            for elem in node.elts:
                if isinstance(elem, ast.Name):
                    # Try to resolve variable
                    var_value = self._find_variable_definition(elem.id, tree, call_line)
                    if var_value:
                        return var_value, "medium", "variable_in_list", f"Variable '{elem.id}' in contents list"
                elif isinstance(elem, ast.Constant) and isinstance(elem.value, str):
                    # Direct string in list
                    return elem.value, "high", "literal_in_list", "String literal in contents list"

            return None, "none", "list", "Could not extract prompt from list"

        # Case 4: Variable reference (medium confidence if we can find definition)
        if isinstance(node, ast.Name):
            var_value = self._find_variable_definition(node.id, tree, call_line)
            if var_value:
                return var_value, "medium", "variable", f"Variable '{node.id}' defined in same scope"
            return None, "low", "variable", f"Variable '{node.id}' - definition not found in static analysis"

        # Case 5: Attribute access (e.g., self.prompt, config.system_prompt)
        if isinstance(node, ast.Attribute):
            attr_path = self._get_attribute_path(node)
            return None, "none", "attribute", f"Attribute access '{attr_path}' - runtime value"

        # Case 6: Function call (runtime value)
        if isinstance(node, ast.Call):
            func_name = self._get_call_name(node)
            return None, "none", "function_call", f"Function call '{func_name}' - runtime value"

        # Case 7: Binary operation (string concatenation, etc.)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                left_val = self._extract_value_from_node(node.left, content, tree, call_line)
                right_val = self._extract_value_from_node(node.right, content, tree, call_line)
                if left_val[0] and right_val[0]:
                    combined = left_val[0] + right_val[0]
                    return combined, "medium", "concatenation", "String concatenation"
            return None, "low", "binary_op", "Complex binary operation - cannot statically evaluate"

        # Case 8: Dict (for single message)
        if isinstance(node, ast.Dict):
            msg_dict = self._extract_dict_as_message(node, content, tree, call_line)
            if msg_dict:
                return msg_dict, "medium", "message_dict", "Single message dictionary"

        # Unknown node type
        node_type = type(node).__name__
        return None, "none", "unknown", f"Unsupported node type: {node_type}"

    def _reconstruct_fstring(self, node: ast.JoinedStr) -> str:
        """Reconstruct f-string with placeholders for variables."""
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                # Get variable name if possible
                if isinstance(value.value, ast.Name):
                    parts.append(f"{{{value.value.id}}}")
                elif isinstance(value.value, ast.Attribute):
                    parts.append(f"{{{self._get_attribute_path(value.value)}}}")
                else:
                    parts.append("{...}")
        return "".join(parts)

    def _extract_messages_list(
        self, node: ast.List, content: str, tree: ast.AST, call_line: int
    ) -> str | None:
        """Extract messages from a list of message dictionaries."""
        messages = []
        for elem in node.elts:
            if isinstance(elem, ast.Dict):
                msg_dict = self._extract_dict_as_message(elem, content, tree, call_line)
                if msg_dict:
                    messages.append(msg_dict)

        if messages:
            return "\n".join(messages)
        return None

    def _extract_dict_as_message(
        self, node: ast.Dict, content: str, tree: ast.AST, call_line: int
    ) -> str | None:
        """Extract a message dictionary as a string."""
        role = None
        content_val = None

        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant):
                if key.value == "role" and isinstance(value, ast.Constant):
                    role = value.value
                elif key.value == "content":
                    extracted = self._extract_value_from_node(value, content, tree, call_line)
                    if extracted[0]:
                        content_val = extracted[0]

        if role and content_val:
            return f"[{role}]: {content_val}"
        elif content_val:
            return content_val
        return None

    def _find_variable_definition(self, var_name: str, tree: ast.AST, before_line: int) -> str | None:
        """
        Find variable definition in the AST before the given line.

        Args:
            var_name: Variable name to find
            tree: AST tree
            before_line: Only look at definitions before this line

        Returns:
            The string value if found, None otherwise
        """
        class VarFinder(ast.NodeVisitor):
            def __init__(self):
                self.value = None
                self.found = False

            def visit_Assign(self, node):
                if self.found or node.lineno >= before_line:
                    return

                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        # Found the variable assignment
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            self.value = node.value.value
                            self.found = True
                        elif isinstance(node.value, ast.JoinedStr):
                            try:
                                self.value = self._reconstruct_fstring(node.value)
                                self.found = True
                            except:
                                pass
                        elif isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mod):
                            # Handle % string formatting: "template %s" % var
                            try:
                                if isinstance(node.value.left, ast.Constant) and isinstance(node.value.left.value, str):
                                    # Get the template string and show with placeholder
                                    template = node.value.left.value
                                    # Replace all format specifiers with placeholders
                                    # Handles: %s, %d, %f, %(name)s, etc.
                                    import re as regex_module
                                    template_with_placeholders = regex_module.sub(
                                        r'%(?:\([^)]+\))?[sdifrgxXoeEfFgGcrabu%]',
                                        '{...}',
                                        template
                                    )
                                    self.value = template_with_placeholders
                                    self.found = True
                            except:
                                pass
                self.generic_visit(node)

            def _reconstruct_fstring(self, node: ast.JoinedStr) -> str:
                """Reconstruct f-string with placeholders."""
                parts = []
                for value in node.values:
                    if isinstance(value, ast.Constant):
                        parts.append(str(value.value))
                    elif isinstance(value, ast.FormattedValue):
                        if isinstance(value.value, ast.Name):
                            parts.append(f"{{{value.value.id}}}")
                        else:
                            parts.append("{...}")
                return "".join(parts)

        finder = VarFinder()
        finder.visit(tree)
        return finder.value

    def _get_attribute_path(self, node: ast.Attribute) -> str:
        """Get full attribute path (e.g., 'self.config.prompt')."""
        parts = [node.attr]
        current = node.value
        while isinstance(current, ast.Attribute):
            parts.insert(0, current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.insert(0, current.id)
        return ".".join(parts)

    def _get_call_name(self, node: ast.Call) -> str:
        """Get the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return self._get_attribute_path(node.func)
        return "<unknown>"

    def find_llm_calls_in_directory(
        self, directory: Path, exclude_patterns: List[str] | None = None
    ) -> List[LLMCallInstance]:
        """
        Find all LLM API calls in a directory tree.

        Args:
            directory: Root directory to search
            exclude_patterns: List of glob patterns to exclude (e.g., ['*/test/*', '*/venv/*'])

        Returns:
            List of all found LLM call instances
        """
        exclude_patterns = exclude_patterns or ["*/venv/*", "*/env/*", "*/.venv/*", "*/node_modules/*"]

        all_instances = []

        for py_file in directory.rglob("*.py"):
            # Check if file should be excluded
            if any(py_file.match(pattern) for pattern in exclude_patterns):
                logger.debug("file_excluded", file=str(py_file))
                continue

            instances = self.find_llm_calls_in_file(py_file)
            all_instances.extend(instances)

        logger.info(
            "directory_analyzed",
            directory=str(directory),
            total_files=len(list(directory.rglob("*.py"))),
            total_instances=len(all_instances),
        )

        return all_instances

    def generate_report(self, instances: List[LLMCallInstance]) -> Dict[str, Any]:
        """
        Generate a summary report of found LLM calls.

        Args:
            instances: List of LLM call instances

        Returns:
            Dictionary with summary statistics
        """
        report = {
            "total_calls": len(instances),
            "by_provider": {},
            "by_library": {},
            "by_call_type": {},
            "by_file": {},
            "providers_used": set(),
            "libraries_used": set(),
        }

        for instance in instances:
            # Count by provider
            report["by_provider"][instance.provider] = (
                report["by_provider"].get(instance.provider, 0) + 1
            )
            report["providers_used"].add(instance.provider)

            # Count by library
            report["by_library"][instance.library] = (
                report["by_library"].get(instance.library, 0) + 1
            )
            report["libraries_used"].add(instance.library)

            # Count by call type
            report["by_call_type"][instance.call_type] = (
                report["by_call_type"].get(instance.call_type, 0) + 1
            )

            # Count by file
            report["by_file"][instance.file_path] = (
                report["by_file"].get(instance.file_path, 0) + 1
            )

        # Convert sets to lists for JSON serialization
        report["providers_used"] = sorted(list(report["providers_used"]))
        report["libraries_used"] = sorted(list(report["libraries_used"]))

        return report

    def export_to_json(self, instances: List[LLMCallInstance], output_file: Path) -> None:
        """
        Export found instances to JSON file.

        Args:
            instances: List of instances to export
            output_file: Path to output JSON file
        """
        import json

        data = {
            "summary": self.generate_report(instances),
            "instances": [
                {
                    "file": inst.file_path,
                    "line": inst.line_number,
                    "provider": inst.provider,
                    "library": inst.library,
                    "call_type": inst.call_type,
                    "code": inst.code_snippet,
                    "function": inst.function_name,
                    "confidence": inst.confidence,
                    "prompt": {
                        "extracted": inst.extracted_prompt,
                        "confidence": inst.prompt_confidence,
                        "method": inst.extraction_method,
                        "notes": inst.extraction_notes,
                    },
                }
                for inst in instances
            ],
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("exported_to_json", file=str(output_file), instances=len(instances))
