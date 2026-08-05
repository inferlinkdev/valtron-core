# Install

Valtron requires Python 3.12 through 3.14. Pick your operating system and package manager below to see the exact commands.

<div class="install-picker">
  <div class="install-picker-row">
    <span class="install-picker-label">Operating System</span>
    <button class="install-picker-btn is-active" data-group="os" data-value="windows">Windows</button>
    <button class="install-picker-btn" data-group="os" data-value="macos">macOS</button>
    <button class="install-picker-btn" data-group="os" data-value="linux">Linux</button>
  </div>
  <div class="install-picker-row">
    <span class="install-picker-label">Package Manager</span>
    <button class="install-picker-btn is-active" data-group="pm" data-value="pip">pip</button>
    <button class="install-picker-btn" data-group="pm" data-value="poetry">poetry</button>
    <button class="install-picker-btn" data-group="pm" data-value="uv">uv</button>
  </div>
</div>

<div class="install-picker-os-note" data-os="windows">Install the 64-bit version of Python 3 from the <a href="https://www.python.org/downloads/">official Python website</a>. During setup, check "Add python.exe to PATH" so the commands below work from any terminal.</div>
<div class="install-picker-os-note" data-os="macos" hidden>Install Python 3 with Homebrew (<code>brew install python</code>) or from the <a href="https://www.python.org/downloads/">official Python website</a>.</div>
<div class="install-picker-os-note" data-os="linux" hidden>Python 3 is usually installed by default. Check with <code>python3 --version</code>; if it's missing, install <code>python3</code> and <code>python3-pip</code> from your distribution's package manager (for example, <code>apt install python3 python3-pip</code> on Debian/Ubuntu).</div>

<pre class="install-picker-content" data-os="windows" data-pm="pip">C:\&gt; python -m venv valtron-env
C:\&gt; valtron-env\Scripts\activate
C:\&gt; pip install -U valtron-core</pre>
<pre class="install-picker-content" data-os="windows" data-pm="poetry" hidden>C:\&gt; poetry add valtron-core</pre>
<pre class="install-picker-content" data-os="windows" data-pm="uv" hidden>C:\&gt; uv add valtron-core</pre>
<pre class="install-picker-content" data-os="macos" data-pm="pip" hidden>$ python -m venv valtron-env
$ source valtron-env/bin/activate
$ pip install -U valtron-core</pre>
<pre class="install-picker-content" data-os="macos" data-pm="poetry" hidden>$ poetry add valtron-core</pre>
<pre class="install-picker-content" data-os="macos" data-pm="uv" hidden>$ uv add valtron-core</pre>
<pre class="install-picker-content" data-os="linux" data-pm="pip" hidden># python3 -m venv valtron-env
# source valtron-env/bin/activate
# pip3 install -U valtron-core</pre>
<pre class="install-picker-content" data-os="linux" data-pm="poetry" hidden># poetry add valtron-core</pre>
<pre class="install-picker-content" data-os="linux" data-pm="uv" hidden># uv add valtron-core</pre>

<div class="install-picker-pm-note" data-pm="pip">The virtual environment step above is optional, but it is strongly recommended because it avoids conflicts with other installed packages.</div>
<div class="install-picker-pm-note" data-pm="poetry" hidden>Poetry creates and manages its own virtual environment automatically, so there is no separate environment step to run.</div>
<div class="install-picker-pm-note" data-pm="uv" hidden>uv creates and manages its own virtual environment automatically, so there is no separate environment step to run.</div>

<script>
(function () {
  var state = { os: "windows", pm: "pip" };

  function render() {
    document.querySelectorAll(".install-picker-content").forEach(function (el) {
      el.hidden = !(el.dataset.os === state.os && el.dataset.pm === state.pm);
    });
    document.querySelectorAll(".install-picker-os-note").forEach(function (el) {
      el.hidden = el.dataset.os !== state.os;
    });
    document.querySelectorAll(".install-picker-pm-note").forEach(function (el) {
      el.hidden = el.dataset.pm !== state.pm;
    });
  }

  document.querySelectorAll(".install-picker-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var group = btn.dataset.group;
      state[group] = btn.dataset.value;
      document.querySelectorAll('.install-picker-btn[data-group="' + group + '"]').forEach(function (b) {
        b.classList.toggle("is-active", b === btn);
      });
      render();
    });
  });
})();
</script>

## Verify your installation

Run the following to confirm both the package and an API key are working:

```python
from valtron_core.client import LLMClient
import asyncio

async def test():
    client = LLMClient()
    response = await client.complete(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
    )
    print(response.choices[0].message.content)

asyncio.run(test())
```

This requires an API key for at least one LLM provider, set as an environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) or in a `.env` file in your working directory. See the [Config Format](../user-guide/classification/config-format.md) guide and your provider's documentation for the full list of supported variables.

## Optional extras

Training or running inference on local transformer models, such as DistilBERT, requires the `transformers` extra:

```bash
pip install "valtron-core[transformers]"
```

This adds `torch`, `transformers`, `scikit-learn`, `datasets`, and `accelerate` to your environment. See [Transformer Models](../user-guide/self-hosting/transformer-models) for how to use them.

## Building from source

Building from source is mainly needed if you want to contribute to the project, since it installs an editable copy from a local clone rather than a published release:

```bash
git clone https://github.com/inferlinkdev/valtron-core.git
cd valtron-core
poetry install --extras transformers
```

See the [repository on GitHub](https://github.com/inferlinkdev/valtron-core) for more details.

## Dependencies

The table below lists Valtron's runtime dependencies and what each one is used for.

| Dependency | Minimum version | Purpose |
|---|---|---|
| pydantic | 2.0 | Config and data models |
| pydantic-settings | 2.0 | Environment-based settings |
| litellm | 1.0.0 | Universal LLM provider interface |
| python-dotenv | 1.0.0 | `.env` file loading |
| structlog | 24.0.0 | Structured logging |
| typer | 0.12 | CLI entry points |
| rich | 13.0.0 | CLI output formatting |
| tqdm | 4.0 | Progress bars |
| rapidfuzz | 3.14.3 | Fuzzy text similarity, see [Field Metrics](../user-guide/extraction/field-metrics/leaf-fields) |
| scipy | 1.13 | Optimal list-item alignment |
| requests | 2.31.0 | HTTP calls for attachments and the configuration wizard |
| matplotlib | 3.8.0 | PDF report charts |
| jinja2 | 3.1.0 | HTML report templates |
| flask / flask-cors | 3.0.0 / 4.0.0 | The [configuration wizard](./configuration-wizard) UI |
| nltk | 3.9.2 | Text similarity metrics |
| reportlab | 4.4.10 | PDF report generation |
| jsonschema | 4.26.0 | Response schema validation |
| torch, transformers, scikit-learn, datasets, accelerate | n/a | Only required by the `transformers` extra described above |
