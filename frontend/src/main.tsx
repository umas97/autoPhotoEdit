// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { MutationCache, QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { App } from './App'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Toasts } from './components/Toasts'
import { toast } from './lib/toast'
import './index.css'

// Everything is on loopback, so a request costs a millisecond and a refetch is
// not something to economise on. What matters is not showing stale counts: the
// progress socket already pushes the truth four times a second, and a query
// that refuses to refetch would contradict it.
//
// A mutation that fails and handles nothing itself says so in a toast: a save
// that did not happen must never look like one that did. Those with their own
// onError (the review's notices, the export's conflicts) keep their own words.
const queryClient = new QueryClient({
  mutationCache: new MutationCache({
    onError: (error, _variables, _context, mutation) => {
      if (mutation.options.onError) return
      if (error instanceof DOMException && error.name === 'AbortError') return
      toast(error.message)
    },
  }),
  defaultOptions: {
    queries: {
      staleTime: 2_000,
      retry: 1,
      refetchOnWindowFocus: true,
    },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <App />
          <Toasts />
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  </StrictMode>,
)
