"""Exercise the user-facing stream interface independently of fixture generation."""
import subprocess
import sys
import unittest


class CommandLineTests(unittest.TestCase):
    def run_cli(self, source, target, data):
        return subprocess.run(
            [sys.executable, "-m", "sylang_core", "--from", source, "--to", target],
            input=data, capture_output=True, check=False,
        )

    def test_unicode_stream_conversion_and_back(self):
        original = 'M0.1:p("Zoë",s,"東京",-,p,c,r)'.encode("utf-8")
        first = self.run_cli("m", "prime", original)
        self.assertEqual(first.returncode, 0, first.stderr)
        second = self.run_cli("prime", "m", first.stdout)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(second.stdout.rstrip(b"\n"), original)

    def test_bad_document_fails_without_output(self):
        result = self.run_cli("m", "json", b'M0.1:p("A",unknown,"B",+,n,s,u)')
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout)
        self.assertIn(b"sylang:", result.stderr)

    def test_invalid_utf8_fails_without_traceback(self):
        result = self.run_cli("m", "json", b"\xff")
        self.assertEqual(result.returncode, 2)
        self.assertNotIn(b"Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
