"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactNode, useState } from "react";
import { RegionProvider } from "@/components/providers/RegionProvider";
import { DevRefProvider } from "@/components/providers/DevRefProvider";

interface ProvidersProps {
  children: ReactNode;
}

export function Providers({ children }: ProvidersProps) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000, // 1 minute
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <DevRefProvider>
        <RegionProvider>{children}</RegionProvider>
      </DevRefProvider>
    </QueryClientProvider>
  );
}
