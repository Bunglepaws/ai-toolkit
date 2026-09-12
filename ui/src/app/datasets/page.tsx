'use client';

import { useEffect, useMemo, useState } from 'react';
import { Modal } from '@/components/Modal';
import Link from 'next/link';
import { TextInput } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import useDatasetStats from '@/hooks/useDatasetStats';
import { Button } from '@headlessui/react';
import { FaRegTrashAlt, FaSortAmountDown, FaSortAmountUpAlt } from 'react-icons/fa';
import { openConfirm } from '@/components/ConfirmModal';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { apiClient } from '@/utils/api';
import { useRouter } from 'next/navigation';
import classNames from 'classnames';

type SortKey = 'name' | 'modified' | 'count';

const SORT_KEYS: SortKey[] = ['name', 'modified', 'count'];
// Remembered per browser so the page comes back sorted the way it was left.
const SORT_STORAGE_KEY = 'aitk_datasets_sort';

function formatDateTime(epochMs: number): string {
  return new Date(epochMs).toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export default function Datasets() {
  const router = useRouter();
  const { datasets, status, refreshDatasets } = useDatasetList();
  const { stats, refreshStats } = useDatasetStats(datasets);
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isNewDatasetModalOpen, setIsNewDatasetModalOpen] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>('name');
  const [sortAsc, setSortAsc] = useState(true);

  // Restore the last sort. Done after mount rather than in the initial state because
  // localStorage doesn't exist during server rendering.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(SORT_STORAGE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw);
      if (SORT_KEYS.includes(saved?.key)) {
        setSortKey(saved.key);
        setSortAsc(saved.asc !== false);
      }
    } catch {
      // unreadable/corrupt preference - fall back to the default sort
    }
  }, []);

  const applySort = (key: SortKey, asc: boolean) => {
    setSortKey(key);
    setSortAsc(asc);
    try {
      localStorage.setItem(SORT_STORAGE_KEY, JSON.stringify({ key, asc }));
    } catch {
      // private mode / storage disabled - preference just doesn't persist
    }
  };

  // Clicking a header sorts by it; clicking it again flips the direction.
  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      applySort(key, !sortAsc);
    } else {
      // Names read best A-Z; for dates and counts the interesting end is the big one.
      applySort(key, key === 'name');
    }
  };

  const sortHeader = (title: string, key: SortKey, alignRight = false) => (
    <button
      type="button"
      onClick={() => toggleSort(key)}
      className={classNames(
        'flex items-center gap-1 uppercase hover:text-gray-200 transition-colors',
        alignRight ? 'justify-end w-full' : '',
      )}
    >
      <span>{title}</span>
      {sortKey === key && (sortAsc ? <FaSortAmountUpAlt size={11} /> : <FaSortAmountDown size={11} />)}
    </button>
  );

  // Transform datasets array into rows with objects. Stats arrive after the names, so a
  // row may not have them yet.
  const tableRows = useMemo(() => {
    const rows = datasets.map(dataset => ({
      name: dataset,
      modified: stats[dataset]?.modified ?? null,
      count: stats[dataset]?.count ?? null,
      actions: dataset, // Pass full dataset name for actions
    }));

    rows.sort((a, b) => {
      if (sortKey === 'name') {
        return sortAsc ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name);
      }
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      // Rows whose stats haven't loaded yet stay at the bottom either way rather than
      // jumping around as they fill in.
      if (aVal === null && bVal === null) return a.name.localeCompare(b.name);
      if (aVal === null) return 1;
      if (bVal === null) return -1;
      if (aVal === bVal) return a.name.localeCompare(b.name);
      return sortAsc ? aVal - bVal : bVal - aVal;
    });

    return rows;
  }, [datasets, stats, sortKey, sortAsc]);

  const columns: TableColumn[] = [
    {
      title: sortHeader('Dataset Name', 'name'),
      key: 'name',
      render: row => (
        <Link href={`/datasets/${row.name}`} className="text-gray-200 hover:text-gray-100">
          {row.name}
        </Link>
      ),
    },
    {
      title: sortHeader('Modified', 'modified'),
      key: 'modified',
      className: 'w-48 whitespace-nowrap',
      render: row =>
        row.modified === null ? (
          <span className="text-gray-600">--</span>
        ) : (
          <span className="text-gray-400">{formatDateTime(row.modified)}</span>
        ),
    },
    {
      title: sortHeader('Files', 'count', true),
      key: 'count',
      className: 'w-20 text-right',
      render: row =>
        row.count === null ? (
          <span className="text-gray-600">--</span>
        ) : (
          <span className="text-gray-400">{row.count.toLocaleString()}</span>
        ),
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'w-20 text-right',
      render: row => (
        <button
          className="text-gray-200 hover:bg-red-600 p-2 rounded-full transition-colors"
          onClick={() => handleDeleteDataset(row.name)}
        >
          <FaRegTrashAlt />
        </button>
      ),
    },
  ];

  const handleDeleteDataset = (datasetName: string) => {
    openConfirm({
      title: 'Delete Dataset',
      message: `Are you sure you want to delete the dataset "${datasetName}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/datasets/delete', { name: datasetName })
          .then(() => {
            console.log('Dataset deleted:', datasetName);
            refreshDatasets();
          })
          .catch(error => {
            console.error('Error deleting dataset:', error);
          });
      },
    });
  };

  const handleCreateDataset = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const data = await apiClient.post('/api/datasets/create', { name: newDatasetName }).then(res => res.data);
      console.log('New dataset created:', data);
      refreshDatasets();
      setNewDatasetName('');
      setIsNewDatasetModalOpen(false);
    } catch (error) {
      console.error('Error creating new dataset:', error);
    }
  };

  const openNewDatasetModal = () => {
    openConfirm({
      title: 'New Dataset',
      message: 'Enter the name of the new dataset:',
      type: 'info',
      confirmText: 'Create',
      inputTitle: 'Dataset Name',
      onConfirm: async (name?: string) => {
        if (!name) {
          console.error('Dataset name is required.');
          return;
        }
        try {
          const data = await apiClient.post('/api/datasets/create', { name }).then(res => res.data);
          console.log('New dataset created:', data);
          if (data.name) {
            router.push(`/datasets/${data.name}`);
          } else {
            refreshDatasets();
          }
        } catch (error) {
          console.error('Error creating new dataset:', error);
        }
      },
    });
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">Datasets</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Button
            className="text-white bg-slate-600 px-2 sm:px-3 py-1 rounded-md hover:bg-slate-500 transition-colors text-sm sm:text-base whitespace-nowrap"
            onClick={() => openNewDatasetModal()}
          >
            <span className="sm:hidden">+ New</span>
            <span className="hidden sm:inline">New Dataset</span>
          </Button>
        </div>
      </TopBar>

      <MainContent>
        <UniversalTable
          columns={columns}
          rows={tableRows}
          isLoading={status === 'loading'}
          onRefresh={() => {
            refreshDatasets();
            refreshStats();
          }}
        />
      </MainContent>

      <Modal
        isOpen={isNewDatasetModalOpen}
        onClose={() => setIsNewDatasetModalOpen(false)}
        title="New Dataset"
        size="md"
      >
        <div className="space-y-4 text-gray-200">
          <form onSubmit={handleCreateDataset}>
            <div className="text-sm text-gray-400">
              This will create a new folder with the name below in your dataset folder.
            </div>
            <div className="mt-4">
              <TextInput label="Dataset Name" value={newDatasetName} onChange={value => setNewDatasetName(value)} />
            </div>

            <div className="mt-6 flex justify-end space-x-3">
              <button
                type="button"
                className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
                onClick={() => setIsNewDatasetModalOpen(false)}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                Confirm
              </button>
            </div>
          </form>
        </div>
      </Modal>
    </>
  );
}
