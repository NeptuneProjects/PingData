
import os, sys

# Add 'pingmapper' to the path, may not need after pypi package...
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PACKAGE_DIR)

# Default function to run
if len(sys.argv) == 1:
    to_do = 'gui_main'
else:
    to_do = sys.argv[1]

# May need to load conda config, check later

def main(process):
    
    # Process single sonar log
    if process == 'gui_main':
        from pingmapper.gui_main import main
        main()

    # Do test on small dataset
    elif process == 'test':
        print('\n\nTesting PINGMapper on small dataset...\n\n')
        from pingmapper.test_PINGMapper import test
        test(1)

    # Do test on large dataset
    elif process == 'test_large':
        print('\n\nTesting PINGMapper on large dataset...\n\n')
        from pingmapper.test_PINGMapper import test
        test(2)

if __name__ == "__main__":
    main(to_do)