import { app } from '../../../scripts/app.js'

// Opens a saved vidAI workflow in the canvas when the page is loaded with
// ?vidai_wf=<name>. The workflow JSON is looked up in two places:
//   /ui/workflows/<name>.ui.json      (nginx static, public host)
//   /view?filename=<name>.ui.json     (ComfyUI input dir, localhost)
async function fetchWorkflow(name) {
    const candidates = [
        `/ui/workflows/${encodeURIComponent(name)}.ui.json`,
        `/view?filename=${encodeURIComponent(name)}.ui.json&type=input`,
    ]
    for (const url of candidates) {
        try {
            const res = await fetch(url)
            if (!res.ok) continue
            return await res.json()
        } catch (err) {
            console.warn(`vidAI: ${url} failed:`, err)
        }
    }
    throw new Error('workflow file not found')
}

app.registerExtension({
    name: 'vidAI.WorkflowLoader',
    async setup() {
        const name = new URLSearchParams(window.location.search).get('vidai_wf')
        if (!name) return

        try {
            const graph = await fetchWorkflow(name)
            await app.loadGraphData(graph, true, true)
            app.canvas.setDirty(true, true)
            console.log(`vidAI: loaded workflow "${name}"`)
        } catch (err) {
            console.error(`vidAI: could not load workflow "${name}":`, err)
            alert(`vidAI: could not load workflow "${name}": ${err.message}`)
        }
    },
})
