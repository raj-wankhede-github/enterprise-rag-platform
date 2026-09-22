/**
 * The Ask page, and chiefly how it renders a refusal.
 *
 * `answerable: false` is a successful outcome. If the UI styles it as an error, users learn that
 * the one honest thing the system does is a malfunction — and the pressure that follows is to
 * make it answer anyway, which is the failure mode the whole product is built to avoid.
 *
 * The other thing tested here is that document text is never rendered as markup. A retrieved
 * passage is untrusted content, and a remote image in it is the classic exfiltration channel.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Ask from "./Ask";
import type { AnswerResponse } from "@/api/client";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

const ANSWERED: AnswerResponse = {
  answerable: true,
  answer: "The per diem is 120 EUR for grade A.",
  claims: [{ text: "The per diem is 120 EUR for grade A.", evidence_ids: ["c7f3a1b2"] }],
  citations: [
    {
      chunk_id: "c7f3a1b2",
      doc_id: "doc-1",
      title: "Travel & Expense Policy v4",
      heading_path: "Reimbursement > Per Diem",
      page_from: 12,
      snippet: "Employees of grade A may claim 120 EUR per night without receipts.",
    },
  ],
  abstention_reason: null,
  sub_questions_answered: [],
  diagnostics: {},
};

const ABSTAINED: AnswerResponse = {
  answerable: false,
  answer:
    "I could not find an answer to this in the documents available to you. I searched the travel and expense policies and found nothing covering overnight allowances for contractors.",
  claims: [],
  citations: [],
  abstention_reason: "NO_SUPPORTING_EVIDENCE",
  sub_questions_answered: [],
  diagnostics: {},
};

async function ask(response: AnswerResponse | (() => Response)) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
    typeof response === "function" ? response() : jsonResponse(200, response),
  );
  render(<Ask />);
  await userEvent.type(screen.getByLabelText("Question"), "What is the per diem?");
  await userEvent.click(screen.getByRole("button", { name: "Ask" }));
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("an answer", () => {
  it("renders the claim and a clickable citation", async () => {
    await ask(ANSWERED);

    expect(await screen.findByText(/120 EUR for grade A/)).toBeInTheDocument();
    const marker = screen.getByRole("link", { name: "c7f3a1" });
    expect(marker).toHaveAttribute("href", "#source-c7f3a1b2");
  });

  it("lists the source with its document, section and page", async () => {
    await ask(ANSWERED);

    expect(await screen.findByText("Travel & Expense Policy v4")).toBeInTheDocument();
    expect(screen.getByText("Reimbursement > Per Diem")).toBeInTheDocument();
    expect(screen.getByText(/p\.\s*12/)).toBeInTheDocument();
  });

  it("anchors each source so a citation click lands on it", async () => {
    await ask(ANSWERED);
    await screen.findByText("Travel & Expense Policy v4");
    expect(document.getElementById("source-c7f3a1b2")).not.toBeNull();
  });
});

describe("a refusal", () => {
  it("is not rendered as an error", async () => {
    // The whole point. An alert role, or red styling, teaches users that honesty is a fault.
    await ask(ABSTAINED);

    expect(await screen.findByText(/could not find an answer/i)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("says what was searched rather than only that it failed", async () => {
    await ask(ABSTAINED);
    expect(await screen.findByText(/searched the travel and expense policies/i)).toBeInTheDocument();
  });

  it("shows the sub-questions that were answerable rather than discarding them", async () => {
    await ask({
      ...ABSTAINED,
      sub_questions_answered: ["What is the standard per diem rate?"],
    });

    expect(await screen.findByText("Partly answered")).toBeInTheDocument();
    expect(screen.getByText("What is the standard per diem rate?")).toBeInTheDocument();
  });

  it("offers the nearest material when there is any, and labels it honestly", async () => {
    await ask({ ...ABSTAINED, citations: ANSWERED.citations });

    expect(await screen.findByText(/did not support an answer/i)).toBeInTheDocument();
    expect(screen.getByText("Travel & Expense Policy v4")).toBeInTheDocument();
  });

  it("shows no citations section when there is nothing close", async () => {
    await ask(ABSTAINED);
    await screen.findByText(/could not find an answer/i);
    expect(screen.queryByRole("region", { name: "Sources" })).not.toBeInTheDocument();
  });
});

describe("untrusted document content", () => {
  it("renders a passage as text, never as markup", async () => {
    // A markdown image in a retrieved passage is the classic exfiltration channel: the renderer
    // fetching it would send the surrounding context to whoever wrote the document.
    await ask({
      ...ANSWERED,
      citations: [
        {
          ...ANSWERED.citations[0]!,
          snippet: '<img src="https://attacker.example.com/x.png"> and <script>alert(1)</script>',
        },
      ],
    });

    await screen.findByText(/attacker\.example\.com/);
    expect(document.querySelector("img")).toBeNull();
    expect(document.querySelector("script")).toBeNull();
  });
});

describe("failures that really are failures", () => {
  it("shows a server error as an alert", async () => {
    await ask(() => jsonResponse(503, { code: "unavailable", message: "Search is temporarily unavailable." }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Search is temporarily unavailable.");
  });

  it("does not submit an empty question", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    render(<Ask />);
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
