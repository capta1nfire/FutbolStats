"use client";

import { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { FileText, Copy, Check, Loader2, AlertTriangle } from "lucide-react";
import { parsePromptTemplate, PromptTemplateResponse } from "@/lib/api/settings";
import { toast } from "sonner";

export function PromptTemplateViewer() {
  const [data, setData] = useState<PromptTemplateResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    async function fetchTemplate() {
      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch("/api/settings/ia-features/prompt-template");
        if (!response.ok) {
          throw new Error(`Failed to fetch: ${response.status}`);
        }

        const json = await response.json();
        const parsed = parsePromptTemplate(json);

        if (parsed) {
          setData(parsed);
        } else {
          throw new Error("Invalid response format");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setIsLoading(false);
      }
    }

    fetchTemplate();
  }, []);

  const handleCopy = async () => {
    if (!data) return;

    try {
      await navigator.clipboard.writeText(data.promptTemplate);
      setCopied(true);
      toast.success("Prompt copied to clipboard");
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast.error("Failed to copy");
    }
  };

  if (isLoading) {
    return (
      <div className="bg-surface rounded-lg p-4">
        <div className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          <span className="text-sm text-muted-foreground">Loading prompt template...</span>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-surface rounded-lg p-4">
        <div className="flex items-center gap-2 text-[var(--status-warning-text)]">
          <AlertTriangle className="h-4 w-4" />
          <span className="text-sm">Failed to load prompt template</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-surface rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Prompt Template</span>
          <Badge variant="outline" className="text-xs">
            {data.version}
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">
            {data.charCount.toLocaleString()} chars
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="h-7 px-2"
          >
            {copied ? (
              <Check className="h-3.5 w-3.5 text-[var(--status-success-text)]" />
            ) : (
              <Copy className="h-3.5 w-3.5" />
            )}
          </Button>
        </div>
      </div>

      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="prompt" className="border-none">
          <AccordionTrigger className="py-2 hover:no-underline text-sm">
            Ver prompt completo
          </AccordionTrigger>
          <AccordionContent>
            <pre className="text-xs bg-muted/50 rounded-md p-3 overflow-x-auto max-h-[400px] overflow-y-auto whitespace-pre-wrap font-mono">
              {data.promptTemplate}
            </pre>
            <p className="text-xs text-muted-foreground mt-2">{data.notes}</p>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}
