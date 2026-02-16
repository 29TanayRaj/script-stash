# Pytest Tutorial

Pytest is a testing framework for Python designed to help you write better code with less effort:

*   **Simple Syntax**: Uses plain Python `assert` statements, making tests readable and easy to write.
*   **Highly Scalable**: Works efficiently for small scripts and large-scale applications.
*   **Rich Ecosystem**: Offers a massive library of plugins for various specialized testing needs.
---

## 1. Test Discovery and Naming Conventions

For Pytest to automatically find and execute your tests, you must follow specific naming conventions:

*   **Test Files**: Should be named `test_*.py` or `*_test.py`. For example, `test_api.py` or `logic_test.py`.
*   **Test Functions**: Must start with the prefix `test_`. For example, `def test_addition():`.
*   **Test Classes**: Should start with `Test` and should not have an `__init__` method. For example, `class TestCalculator:`.

## 2. Command Line Usage

Pytest provides a powerful CLI to control how your tests run. Here are the most common commands:

### Basic Execution
*   `pytest`: Runs all tests in the current directory and subdirectories that match the naming conventions.
*   `pytest test_api.py`: Runs tests in a specific file.
*   `pytest tests/`: Runs all tests within a specific folder.

### Useful Flags
*   `-v` (Verbose): Shows more detail for each test, including the test names.
*   `-s`: Disables output capturing, allowing you to see `print()` statements in the terminal during test execution.
*   `-x`: Stops the test execution immediately after the first failure.
*   `-k "expression"`: Only runs tests whose names match the expression. For example, `pytest -k "borrow"` runs all tests with "borrow" in their name.
*   `--lf` (Last Failed): Only runs the tests that failed during the last run.

---

## 3. Core Pytest Concepts

### Basic Assertions
In Pytest, any function starts with `test_` is considered a test case. You use the standard Python `assert` statement to check for expected results.
```python
def test_add_basic():
    # We are testing if our add function returns 3 when given 1 and 2
    assert add(1, 2) == 3
```

### Testing for Expected Errors
Sometimes, you need to ensure your code fails correctly when given invalid data. We use `pytest.raises` to verify that a specific exception is thrown.

```python
def test_add_error():
    # This test passes if a TypeError is raised
    with pytest.raises(TypeError):
        add(1, "invalid_input")
```

### Data-Driven Testing (Parametrization)
Instead of writing multiple test functions for the same logic, you can use `@pytest.mark.parametrize` to run one test with different inputs. This is especially useful for edge cases like floating-point math.
```python
@pytest.mark.parametrize("a, b, expected", [
    (10, 20, 30),
    (0.1, 0.2, pytest.approx(0.3)), # approx handles float precision issues
])
def test_add_parameterized(a, b, expected):
    assert add(a, b) == expected
```

### Understanding Fixtures
Fixtures are functions that run before (Setup) and after (Teardown) your tests. They are perfect for preparing databases or cleaning up temporary files. Using `yield` allows you to define the cleanup logic within the same function.
```python
@pytest.fixture
def temp_db():
    # Setup: prepare the resource
    db = {"users": ["Alice", "Bob"]}
    yield db 
    # Teardown: clean up after the test finishes
    db.clear()
```

### Mocking with Monkeypatch
Mocking allows you to replace complex parts of your system (like an external API call) with a simple "mock" version. This ensures your tests are fast and reliable.
```python
def test_mocking(monkeypatch):
    # We replace the real 'get_api_status' with a mock that always returns 'Online'
    monkeypatch.setattr("__main__.get_api_status", lambda: "Online")
    assert get_api_status() == "Online"
```

---

## 4. The Backend Server (main.py)

We have implemented a Library Management System using FastAPI and SQLAlchemy. This server demonstrates how to handle data persistence using an in-memory SQLite database.

### The Technology Stack
* **FastAPI**: Manages the web routes and automatic documentation.
* **SQLAlchemy**: An ORM that allows us to interact with the database using Python objects instead of raw SQL.
* **SQLite (In-Memory)**: A volatile database that lives in RAM, making it perfect for rapid testing and demonstrations.
* **Pydantic**: Handles data validation and transforms database objects into JSON.

### Running the Server
To start the server with the auto-reload feature enabled, run the following command:
```bash
python run.py
```
The server will be available at `http://localhost:8000`.

---

## 5. API Testing (test_api.py)

Once the server is running, we need to verify that our endpoints work correctly. We use the FastAPI `TestClient` to simulate HTTP requests without needing to open a browser.

### Key Testing Goals
1. **Status Codes**: Check if the server returns `200` for success or `404` for missing records.
2. **JSON Integrity**: Ensure the response body contains the correct data fields.
3. **Logic Verification**: For example, when a book is borrowed, its `is_borrowed` status should change from `False` to `True`.

### Example API Test
```python
def test_borrow_and_return():
    # Step 1: Borrow the book
    borrow_res = client.post("/books/1/borrow")
    assert borrow_res.status_code == 200
    assert borrow_res.json()["is_borrowed"] is True

    # Step 2: Return the book
    return_res = client.post("/books/1/return")
    assert return_res.status_code == 200
    assert return_res.json()["is_borrowed"] is False
```

---

## 6. How to Execute the Demo

### Step 1: Install Dependencies
Ensure you have all the required libraries installed:
```bash
pip install -r requirements.txt
```

### Step 2: Run All Tests
To see everything in action, run Pytest from your terminal:
```bash
pytest -v
```

### Step 3: Selectively Run Tests
* For Pytest core concepts: `pytest demo_basic.py -v`
* For API functionality: `pytest test_api.py -v`

---
*End of Tutorial*
