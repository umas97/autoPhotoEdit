// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The routes. They are readable and stable because they are useful in debugging
// (section 10), but nothing is reachable *only* by typing one: every screen is
// also a click away from another, since the window has no address bar.
//
// An unknown path lands on the projects list rather than on a "not found" page:
// in a program with no way to type a URL, an unknown path means a stale link in
// a cached shell, and the right answer is the home screen.
import { Navigate, Route, Routes } from 'react-router-dom'
import { ProjectsPage } from './pages/Projects'
import { ImportPage } from './pages/Import'
import { ViewerPage } from './pages/Viewer'
import { CullingPage } from './pages/Culling'
import { ScenesPage } from './pages/Scenes'
import { StylesPage } from './pages/Styles'
import { ProjectStylePage } from './pages/ProjectStyle'
import { ReviewPage } from './pages/Review'
import { ExportPage } from './pages/Export'
import { ProblemsPage } from './pages/Problems'
import { MergesPage } from './pages/Merges'
import { SettingsPage } from './pages/Settings'
import { ShortcutsHelp } from './components/ShortcutsHelp'

export function App() {
  return (
    <>
      <ShortcutsHelp />
      <Routes>
        <Route path="/" element={<ProjectsPage />} />
        <Route path="/progetti/:projectId/import" element={<ImportPage />} />
        <Route path="/progetti/:projectId/cernita" element={<CullingPage />} />
        <Route path="/progetti/:projectId/fusioni" element={<MergesPage />} />
        <Route path="/progetti/:projectId/foto" element={<ViewerPage />} />
        <Route path="/progetti/:projectId/scene" element={<ScenesPage />} />
        <Route path="/progetti/:projectId/stile" element={<ProjectStylePage />} />
        <Route path="/progetti/:projectId/revisione" element={<ReviewPage />} />
        <Route path="/progetti/:projectId/export" element={<ExportPage />} />
        <Route path="/stili" element={<StylesPage />} />
        <Route path="/problemi" element={<ProblemsPage />} />
        <Route path="/impostazioni" element={<SettingsPage />} />
        <Route path="/stili/:profileId" element={<StylesPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  )
}
