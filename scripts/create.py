import shutil
import sys


def copy_template(template_path, file_path):
    try:
        shutil.copy(template_path, file_path)
        print(f"Template copied successfully. New instance created at: {file_path}")
    except FileNotFoundError:
        print("Error: Template file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/create.py <type> <new_instance_name>")
        sys.exit(1)

    # Parse command line arguments
    args = sys.argv[1:]

    instance_type = args[0]
    new_instance_name = args[1]

    valid_types = ["model", "sequence_model", "data", "migration"]

    if not new_instance_name:
        print("Error: --new_instance_name is required.")
        sys.exit(1)

    if instance_type == "model":
        template_path = "scripts/templates/model_template.py"
        file_path = f"models/{new_instance_name}.py"
    elif instance_type == "sequence_model":
        template_path = "scripts/templates/sequence_model_template.py"
        file_path = f"models/{new_instance_name}.py"
    elif instance_type == "data":
        template_path = "scripts/templates/dataset_template.py"
        file_path = f"data/{new_instance_name}.py"
    elif instance_type == "migration":
        template_path = "scripts/templates/migration_template.py"
        file_path = f"migrations/{new_instance_name}.py"
    else:
        print(f"Error: Invalid instance type. Valid types are: {valid_types}")
        sys.exit(1)

    copy_template(template_path, file_path)
