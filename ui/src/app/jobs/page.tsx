'use client';

import JobsTable from '@/components/JobsTable';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';
import { useSessionFilter } from '@/hooks/useSessionFilter';
import MruTextInput from '@/components/MruTextInput';

export default function Dashboard() {
  const [filter, setFilter] = useSessionFilter('jobs-filter');

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">Queue</h1>
        </div>
        <div className="flex-1 max-w-xl mx-4">
          <MruTextInput
            value={filter}
            onChange={setFilter}
            mruKey="jobs-filter-mru"
            placeholder="Filter by name or model (supports AND, OR)..."
          />
        </div>
        <div>
          <Link
            href="/jobs/new"
            className="text-white bg-slate-600 px-2 sm:px-3 py-1 rounded-md text-sm sm:text-base whitespace-nowrap"
          >
            <span className="sm:hidden">+ New Job</span>
            <span className="hidden sm:inline">New Training Job</span>
          </Link>
        </div>
      </TopBar>
      <MainContent>
        <JobsTable filter={filter} />
      </MainContent>
    </>
  );
}
