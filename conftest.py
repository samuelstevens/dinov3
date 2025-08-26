# conftest.py
import os


def pytest_addoption(parser):
    parser.addoption(
        "--pt-ckpts",
        action="store",
        default=None,
        help="Path to DINOv3 PyTorch checkpoint directory.",
    )
    parser.addoption(
        "--jax-ckpts",
        action="store",
        default=None,
        help="Path to DINOv3 Jax checkpoint directory.",
    )


def pytest_generate_tests(metafunc):
    """
    If a test asks for 'jax_path', parametrize it over files found in --jax-ckpts.
    If the option isn't supplied, the test will be collected with zero params (i.e., skipped).
    """
    if "jax_path" in metafunc.fixturenames:
        root = metafunc.config.getoption("--jax-ckpts")
        if not root:
            fnames = []
        else:
            fnames = os.listdir(root)

        paths = sorted([
            os.path.join(root, fname) for fname in fnames if fname.endswith(".eqx")
        ])

        # produce stable, human-readable IDs
        ids = [os.path.basename(p) for p in paths]

        # If nothing to test, emit zero params (no tests will run for this function)
        metafunc.parametrize("jax_path", paths, ids=ids)

    if "vit_paths" in metafunc.fixturenames:
        pt_root = metafunc.config.getoption("--pt-ckpts")
        jax_root = metafunc.config.getoption("--jax-ckpts")
        if not pt_root or not jax_root:
            metafunc.parametrize("vits", [], ids=[])
            return

        paths, ids = [], []
        for pt_fname in os.listdir(pt_root):
            for jax_fname in os.listdir(jax_root):
                stem, _ = os.path.splitext(jax_fname)
                if stem in pt_fname:
                    paths.append((
                        stem,
                        (
                            os.path.join(pt_root, pt_fname),
                            os.path.join(pt_root, pt_fname),
                            os.path.join(jax_root, jax_fname),
                        ),
                    ))
                    ids.append(stem)

        # If nothing to test, emit zero params (no tests will run for this function)
        metafunc.parametrize("vit_paths", paths, ids=ids)
