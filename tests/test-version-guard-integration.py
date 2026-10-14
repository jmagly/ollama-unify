#!/usr/bin/env python3
"""Optional version guards must be consulted before disruptive upgrade work."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

REPO = Path(sys.argv.pop(1)).resolve()


class VersionGuardIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.helper = self.root / 'version-guard'
        self.calls = self.root / 'calls'
        self.helper.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "'+str(self.calls)+'"\nexit 7\n')
        self.helper.chmod(0o755)
        self.script = self.root / 'ollama-unify.sh'
        self.script.write_text((REPO/'ollama-unify.sh').read_text().replace(
            '/usr/local/sbin/ollama-version-guard', str(self.helper)))
        self.env = os.environ.copy()
        self.env['OLLAMA_UNIFY_ASSUME_YES'] = '1'
        self.env['PATH'] = str(REPO/'tests/fixtures/bin')+':'+self.env['PATH']

    def test_update_is_refused_before_installer_or_service_changes(self):
        result = subprocess.run(['bash', str(self.script), '--update-ollama'],
                                capture_output=True, text=True, env=self.env, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn('permission-gated', result.stdout+result.stderr)
        self.assertNotIn('Quiescing', result.stdout+result.stderr)
        self.assertFalse(self.calls.exists())  # Merely having a guard cannot grant permission.

    def test_manual_reconcile_cannot_cycle_a_guarded_host(self):
        result = subprocess.run(['bash', str(self.script), '--reconcile'],
                                capture_output=True, text=True, env=self.env, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn('managed by the host version guard', result.stdout+result.stderr)
        self.assertNotIn('Reapplying', result.stdout+result.stderr)
        self.assertFalse(self.calls.exists())

    def render_reconciler(self):
        result = subprocess.run(['bash','-c','source "$1"; render_reconcile_helper',
                                 'test',str(self.script)], capture_output=True,text=True,
                                env=self.env,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        # The legacy path must not touch the live host even if reached in a test.
        return result.stdout.replace('/usr/local/share/ollama-unify/state.env',
                                     str(self.root/'absent-state'))

    def test_reconciler_defers_and_propagates_guard_failure(self):
        result = subprocess.run(['bash'],input=self.render_reconciler(),
                                capture_output=True,text=True,env=self.env,timeout=10)
        self.assertEqual(result.returncode,7,result.stderr)
        self.assertEqual(self.calls.read_text(),'ensure\n')
        self.assertNotIn('reconciled;',result.stdout)

    def test_hosts_without_guard_keep_legacy_no_state_noop(self):
        self.helper.unlink()
        result = subprocess.run(['bash'],input=self.render_reconciler(),
                                capture_output=True,text=True,env=self.env,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('nothing to reconcile',result.stdout)
        self.assertFalse(self.calls.exists())


if __name__=='__main__':
    unittest.main()
