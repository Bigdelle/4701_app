export PYTHONPATH=$(pwd)

for test_file in tests/models/test_*.py; do
    echo "Running $test_file..."
    python -m unittest $test_file
done

for test_file in tests/src/data_collection/test_*.py; do
    echo "Running $test_file..."
    python -m unittest $test_file
done