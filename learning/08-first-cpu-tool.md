# 08: Your First CPU Tool

You will serve your own Python function. This lesson does not need the demo
models, so stop the demo server (Ctrl-C in tab A) if you want the memory
back.

## Write the Tool

Create `scratch/lesson_tools.py`:

```python
from pydantic import BaseModel, Field


class Temperature(BaseModel):
    kelvin: float = Field(description="Temperature in kelvin")
    celsius: float = Field(description="Temperature in degrees Celsius")


def celsius_to_kelvin(celsius: float) -> Temperature:
    """Convert a temperature from degrees Celsius to kelvin."""
    return Temperature(kelvin=celsius + 273.15, celsius=celsius)
```

FastMCP builds the input schema from the type hints, the output schema from
the return annotation, and the description from the docstring. All three are
what an agent sees, so write them carefully.

## Write the Server Config

Create `scratch/lesson_server.yml`:

```yaml
search_tool:
  expose: true
tools:
  - lesson_tools.celsius_to_kelvin
```

## Serve It

`tools` entries are imported by their dotted path, so the serving Python
must be able to import `lesson_tools`. `python -m` puts the **current
directory** on `sys.path`, so run the server from `scratch/`:

```shell
cd scratch
../.venv/bin/python -m nomad serve --transport http --port 8282 lesson_server.yml
```

> The `nomad` console script does **not** add the current directory to
> `sys.path`. With it, you would need `PYTHONPATH=.`. Real projects avoid
> this by packaging their tools, as `container/demo` does with `nomad_demo`.

## Call It

With Inspector, using **Streamable HTTP**:

```shell
npx @modelcontextprotocol/inspector http://localhost:8282/mcp
```

From code mode, create `scratch/lesson_08_gateway.yml`:

```yaml
servers:
  mine:
    transport: http
    url: http://localhost:8282/mcp
```

Then create `scratch/lesson_08.py`:

```python
from mcp_tools.mine import celsius_to_kelvin

RESULT = [celsius_to_kelvin(celsius=c) for c in (-273.15, 0, 25, 100)]
```

From the repository root:

```shell
./.venv/bin/python -m nomad cmx -c scratch/lesson_08_gateway.yml -o - scratch/lesson_08.py
```

## Exercises

1. **Name override.** Change the `tools` entry to a mapping:

   ```yaml
   tools:
     - tool: lesson_tools.celsius_to_kelvin
       name: c_to_k
   ```

   Restart the server and update the import.
2. **Factory with `tool_kwargs`.** Write `make_scaler(factor: float)` that
   returns a typed `scale(x: float) -> float` function. Register it with
   `tool: lesson_tools.make_scaler` and `tool_kwargs: {factor: 3}`.
3. **Validation.** Change `celsius` to
   `Annotated[float, Field(ge=-273.15)]` and call it with `-300`.
4. **Silent typo.** Rename `tools:` to `tols:` and restart. The server starts
   without complaint, but your tool is gone. Unknown top-level keys are
   ignored, so check spelling when a setting seems to have no effect.
5. **Logging rule.** Add `print("hi")` to the tool, then serve it over stdio
   with Inspector (`npx @modelcontextprotocol/inspector -- ../.venv/bin/python
   -m nomad serve lesson_server.yml`). Why does this break the protocol?
   Replace it with `logging.getLogger(__name__).info(...)`.

Next: [09 Tests and lint](09-tests-and-lint.md).
