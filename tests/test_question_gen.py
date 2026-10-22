#!/usr/bin/env python3
"""Unit tests for question_gen.py's multi-hop hardening.

check_multihop() is the automated gate that caught the spurious
source_pages failure mode (model writes a single-hop question while
attributing both pages); pick_related_page() is the overlap pairing that
keeps the two pages combinable.
"""

import random
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from question_gen import check_multihop, pick_related_page


def _page(no, text):
    return {"page_no": no, "text": text}


PAGE_A = _page(100, "CALL executes a procedure. If CALL is executed in a "
                    "transaction block, then the called procedure cannot "
                    "execute transaction control statements.")
PAGE_B = _page(200, "Procedures can be invoked with CALL. A procedure "
                    "invoked by CALL runs inside the caller's transaction "
                    "unless it uses transaction control statements.")


def _mhq(**kw):
    q = {"id": "q001", "query_type": "multi_hop",
         "question": "What happens when CALL runs in a transaction block?",
         "expected_answer": ("If CALL is executed in a transaction block, "
                             "then the called procedure cannot execute "
                             "transaction control statements.\n"
                             "A procedure invoked by CALL runs inside the "
                             "caller's transaction unless it uses "
                             "transaction control statements."),
         "source_pages": [100, 200],
         "fact_pages": [100, 200],
         "provenance": "auto"}
    q.update(kw)
    return q


class TestCheckMultihop(unittest.TestCase):
    def setUp(self):
        self.pages = {100: PAGE_A["text"], 200: PAGE_B["text"]}

    def test_genuine_two_page_question_passes(self):
        ok, reason = check_multihop(_mhq(), self.pages)
        self.assertTrue(ok, reason)

    def test_single_fact_line_fails(self):
        q = _mhq(expected_answer="If CALL is executed in a transaction "
                                 "block, then the called procedure cannot "
                                 "execute transaction control statements.",
                 fact_pages=[100])
        ok, reason = check_multihop(q, self.pages)
        self.assertFalse(ok)
        self.assertIn("one fact line per page", reason)

    def test_fact_pages_must_cover_source_pages(self):
        q = _mhq(fact_pages=[100, 100])
        ok, reason = check_multihop(q, self.pages)
        self.assertFalse(ok)
        self.assertIn("must cover", reason)

    def test_nonverbatim_line_fails(self):
        q = _mhq(expected_answer=("If CALL is executed in a transaction "
                                  "block, then the called procedure cannot "
                                  "execute transaction control statements.\n"
                                  "Procedures love pizza on Fridays."))
        ok, reason = check_multihop(q, self.pages)
        self.assertFalse(ok)
        self.assertIn("not found verbatim", reason)

    def test_both_lines_from_one_page_fails(self):
        # the spurious-attribution failure mode: single-hop question,
        # both pages listed
        q = _mhq(
            expected_answer=("If CALL is executed in a transaction block, "
                             "then the called procedure cannot execute "
                             "transaction control statements.\n"
                             "CALL executes a procedure."),
            fact_pages=[100, 200])
        ok, reason = check_multihop(q, self.pages)
        self.assertFalse(ok)  # line 2 is not on p200

    def test_same_page_twice_fails(self):
        q = _mhq(source_pages=[100, 100], fact_pages=[100, 100])
        ok, _ = check_multihop(q, self.pages)
        self.assertFalse(ok)

    def test_non_multihop_skipped(self):
        q = {"query_type": "exact_term", "source_pages": [100],
             "expected_answer": "anything"}
        ok, _ = check_multihop(q, self.pages)
        self.assertTrue(ok)


class TestPickRelatedPage(unittest.TestCase):
    def test_picks_overlapping_page(self):
        pages = [_page(1, "the quick brown fox jumps over the lazy dog"),
                 _page(2, "completely unrelated zebra quantum xylophone"),
                 _page(3, "the lazy dog sleeps under the brown table")]
        rng = random.Random(0)
        picked = pick_related_page(pages[0], pages, rng)
        self.assertEqual(picked["page_no"], 3)  # shares fox/dog/brown/lazy

    def test_never_picks_itself(self):
        pages = [_page(1, "alpha beta gamma"),
                 _page(2, "alpha beta delta")]
        rng = random.Random(1)
        for _ in range(20):
            self.assertNotEqual(
                pick_related_page(pages[0], pages, rng)["page_no"], 1)

    def test_fallback_when_nothing_overlaps(self):
        pages = [_page(1, "alpha beta gamma"),
                 _page(2, "zebra quantum xylophone")]
        rng = random.Random(2)
        picked = pick_related_page(pages[0], pages, rng)
        self.assertEqual(picked["page_no"], 2)  # only other page


if __name__ == "__main__":
    unittest.main()
