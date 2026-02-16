import pytest
import time

# 1. Basic Assertions
def add(a, b):
    return a + b

def test_add_basic():
    """Simple assertion check"""
    assert add(1, 2) == 3
    assert add(-1, 1) == 0

# 2. Exception Testing
def test_add_error():
    """Verifying that the code raises the correct exception"""
    with pytest.raises(TypeError):
        add(1, "2")

# 3. Parametrization (Data-Driven Testing)
@pytest.mark.parametrize("a, b, expected", [
    (10, 20, 30),
    (0.1, 0.2, pytest.approx(0.3)), # Using approx for floating point precision!
    (-5, -5, -10),
    (100, 200, 300)
])
def test_add_parameterized(a, b, expected):
    """Running the same test logic with different data sets"""
    assert add(a, b) == expected

# 4. Fixtures with Setup and Teardown
@pytest.fixture
def temp_db():
    # SETUP: Create a resource
    print("\n[Setup] Connecting to temporary database...")
    db = {"users": ["Alice", "Bob"]}
    yield db # This is where the test happens
    # TEARDOWN: Clean up the resource
    print("[Teardown] Closing database connection...")
    db.clear()

def test_db_alice_exists(temp_db):
    """Test using a fixture that handles setup/teardown"""
    assert "Alice" in temp_db["users"]

# 5. Built-in Fixtures (tmp_path)
def test_file_creation(tmp_path):
    """Pytest provides built-in fixtures like tmp_path for file operations"""
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "hello.txt"
    p.write_text("Hello Pytest")
    assert p.read_text() == "Hello Pytest"
    assert len(list(tmp_path.iterdir())) == 1

# 6. Custom Markers
@pytest.mark.slow
def test_slow_operation():
    """You can mark tests and run them selectively using 'pytest -m slow'"""
    # time.sleep(1) # Simulated slow test
    assert True

# 7. Mocking
def get_api_status():
    # In a real app, this would make a network call
    return "Offline"

def test_mocking_with_monkeypatch(monkeypatch):
    def mock_status():
        return "Online"

    monkeypatch.setattr("demo_basic.get_api_status", mock_status)

    assert get_api_status() == "Online"