import cytoscape from "cytoscape";

class MathUtils {
    public static getDistanceBetweenNodes(node1: GraphNode, node2: GraphNode): number {
        const pos1 = node1.getPosition()
        const pos2 = node2.getPosition()

        const dx = pos1.x - pos2.x
        const dy = pos1.y - pos2.y

        return Math.round(this.calculateEuclideanDistance(dx, dy))
    }


    public static calculateEuclideanDistance(value1: number, value2: number): number {
        return Math.sqrt(Math.pow(value1, 2) + Math.pow(value2, 2))
    }


    public static getRandomAngularPosition(): Position {
        const randomDistance = Math.random() * 200
        const randomAngle = Math.random() * 2 * Math.PI
        return this.getAngularPosition(randomAngle, randomDistance)
    }


    public static getRandomAngle(): number {
        return Math.random() * 2 * Math.PI
    }


    public static getAngularPosition(angle: number, distance: number): Position {
        return {
            x: distance * Math.cos(angle),
            y: distance * Math.sin(angle),
        }
    }


    public static getRandomAngularPositionWithDistance(distance: number): Position {
        const randomAngle = Math.random() * 2 * Math.PI
        return this.getAngularPosition(randomAngle, distance)
    }
}


class TextUtils {
    public static getRandomString(chars: number): string {
        const charsList = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        let result = ''
        for (let i = 0; i < chars; i++) {
            result += charsList.charAt(Math.floor(Math.random() * charsList.length))
        }
        return result
    }
}


class ApiUtils {
    public static async postData(endpoint: string, data: any): Promise<any> {
        try {
            const url = 'http://localhost:8080/'
            const response = await fetch(url + endpoint, {
                mode: 'cors',
                method: 'POST',
                headers: {
                'Content-Type': 'application/json',
                'Origin': 'https://localhost:3000',
                'Access-Control-Allow-Origin': 'http://localhost:8080'
                },
                body: JSON.stringify(data),
            })
        
            // Handle the response
            const result = await response.json()
            console.log('Success:', result)
            return result
        } catch (error) {
            console.error('Error:', error)
        }
    }

}



interface EdgeData {
    id: string
    source: string
    target: string
    distance: number
}


class Edge {
    private id: string
    private sourceNode: GraphNode
    private targetNode: GraphNode
    private distance: number

    constructor(sourceNode: GraphNode, targetNode: GraphNode) {
        this.id = "e_" + targetNode.getId()
        this.sourceNode = sourceNode
        this.targetNode = targetNode
        this.distance = MathUtils.getDistanceBetweenNodes(sourceNode, targetNode)
        cy.add(this.toObject())
    }

    public setDistance(distance: number): void {
        const cyEdge = cy.edges(`[source = "${this.sourceNode.getId()}"][target = "${this.targetNode.getId()}"]`)
        cyEdge.data('distance', distance)
        this.distance = distance
    }

    public updateDistance(): void {
        this.setDistance(MathUtils.getDistanceBetweenNodes(this.sourceNode, this.targetNode))
    }

    public getDistance(): number {
        return this.distance
    }

    public remove(): void {
        cy.remove(cy.getElementById(this.id))
    }

    public toObject(): { data: EdgeData } {
        return {
            data: {
                id: this.id,
                source: this.sourceNode.getId(),
                target: this.targetNode.getId(),
                distance: this.distance,
            },
        }
    }
}


type Position = {
    x: number
    y: number
}

enum NodeType {
    central_node,
    outer_node,
}

interface NodeData {
    id: string
    label: string
}

class GraphNode {
    protected id: string
    protected label: string
    protected position: Position
    protected type: NodeType
    
    constructor(id: string, label: string, position: Position, type: NodeType) {
        this.id = id;
        this.label = label;
        this.position = position;
        this.type = type;
    }

    public getId(): string {
        return this.id
    }

    public getPosition(): Position { 
        return this.position
    }

    public setLabel(label: string): void {
        this.label = label
        cy.getElementById(this.id).data('label', label)
    }

    public remove(): void {
        cy.remove(cy.getElementById(this.id))
    }

    public toObject(): { data: NodeData; position: Position } {
        return {
            data: {
                id: this.id,
                label: this.label,
            },
            position: this.position,
        }
    }
}


class CentralNode extends GraphNode {
    constructor(id: string, x: number, y: number) {
        let _id = id
        let _label = id
        let _position = { x, y }
        let _type = NodeType.central_node
        super(_id, _label, _position, _type)
        this.addVisualNodeToInterface()
    }

    private addVisualNodeToInterface(): void {
        cy.add(this.toObject()).addClass(this.type.toString()).lock().ungrabify()
    }
}


class OuterNode extends GraphNode {
    constructor(id: string, distance: number = 0) {
        let _id = id
        let _label = id
        let _position = MathUtils.getAngularPosition(MathUtils.getRandomAngle(), distance);
        let _type = NodeType.outer_node
        super(_id, _label, _position, _type)
        this.addVisualNodeToInterface()
    }

    public setPosition(position: Position): void {
        this.position = position
        this.updateVisualPosition()
    }

    public setPositionFromAngle(angle: number): void {
        this.position = MathUtils.getAngularPosition(angle, this.getDistance())
        this.updateVisualPosition()
    }

    public setPositionFromDistance(distance: number): void {
        this.position = MathUtils.getAngularPosition(MathUtils.getRandomAngle(), distance)
        this.updateVisualPosition()
    }

    public getDistance(): number {
        return MathUtils.calculateEuclideanDistance(this.position.x, this.position.y)
    }

    private updateVisualPosition(): void {
        cy.getElementById(this.id).position(this.position)
    }

    private addVisualNodeToInterface(): void {
        cy.add(this.toObject()).addClass(this.type.toString())
    }
}


class Term {
    protected value: string
    protected node: GraphNode | undefined

    constructor(value: string) {
        this.value = value
    }

    public setLabel(value: string): void {
        this.value = value
        this.node?.setLabel(value)
    }

    public getValue(): string {
        return this.value
    }

    public getNode(): GraphNode | undefined {
        return this.node
    }
}


interface View {
    displayViews(): void
    removeViews(): void
}


class NeighbourTerm extends Term implements View {
    protected node: OuterNode | undefined
    private queryTerm: QueryTerm
    private hops: number
    private nodePosition: Position = { x: 0, y: 0 }
    private edge: Edge | undefined
    // While max distance is always 200, then -> 200 / (user distance) = hopToDistanceRatio
    private hopToDistanceRatio: number = 50

    constructor(queryTerm: QueryTerm, value: string = '', hops: number = 0) {
        super(value)
        this.queryTerm = queryTerm
        this.hops = hops
    }

    public displayViews(): void {
        this.node = new OuterNode(TextUtils.getRandomString(6))
        this.node.setPosition(this.nodePosition)
        this.node.setLabel(this.value)
        if (this.queryTerm.getNode() === undefined) return 
        this.edge = new Edge(this.queryTerm.getNode() as CentralNode, this.node)
    }

    public removeViews(): void {
        this.node?.remove()
        this.edge?.remove()
    }

    public getHops(): number {
        return this.hops
    }

    public setHops(hops: number): void {
        this.hops = hops
        const nodeDistance = this.convertHopsToDistance(hops)
        this.nodePosition = MathUtils.getRandomAngularPositionWithDistance(nodeDistance)
        this.updateNodePosition(this.nodePosition)
    }

    public setPosition(position: Position): void {
        let positionDistance = MathUtils.calculateEuclideanDistance(position.x, position.y)
        const nodeDistance = this.edge?.getDistance() ?? 0

        // Validate position so that it is within the range
        if (this.edge !== undefined && this.node !== undefined ) {
            if (positionDistance < 50.0 || positionDistance > 200.0) {
                let angle = Math.atan2(position.y, position.x)
                let adjustedX = Math.cos(angle) * nodeDistance
                let adjustedY = Math.sin(angle) * nodeDistance
                position.x = adjustedX
                position.y = adjustedY
            }
        }
        
        this.nodePosition = position
        this.hops = this.convertDistanceToHops(nodeDistance)
        this.updateNodePosition(position)
    }

    private convertHopsToDistance(hops: number): number {
        return hops * this.hopToDistanceRatio
    }

    private convertDistanceToHops(distance: number): number {
        return distance / this.hopToDistanceRatio
    }

    private updateNodePosition(position: Position): void {
        this.node?.setPosition(position)
        this.edge?.updateDistance()
    }
}


class QueryTerm extends Term implements View {
    protected node: CentralNode | undefined
    private neighbourTerms: NeighbourTerm[] = []

    constructor(value: string) {
        super(value)
    }

    public displayViews(): void {
        this.node = new CentralNode(this.value, 0, 0)
        for (let neighbourTerm of this.neighbourTerms) {
            neighbourTerm.displayViews()
        }
    }

    /**
    * Removes the views of the neighbour terms and the central node.
    */
    public removeViews(): void {
        for (let neighbourTerm of this.neighbourTerms) {
            neighbourTerm.removeViews()
        }
        this.node?.remove()
    }

    public getNeighbourTerms(): NeighbourTerm[] {
        return this.neighbourTerms
    }

    public getNeighbourTermById(id: string): NeighbourTerm | undefined {
        return this.neighbourTerms.find(p => p.getNode()?.getId() === id)
    }

    public addNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        this.neighbourTerms.push(neighbourTerm)
    }

    public removeNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        this.neighbourTerms = this.neighbourTerms.filter(term => term !== neighbourTerm)
        neighbourTerm.removeViews()
    }
}


class NeighbourTermsTable {
    private activeTermsService: QueryTermService | undefined
    private table: HTMLElement

    constructor() {
        this.table = document.getElementById('neighboursTermsTable') as HTMLElement
    }

    public setActiveService(termsService: QueryTermService): void {
        this.activeTermsService = termsService
    }

    public updateTable(): void {
        const tbody = this.table.getElementsByTagName('tbody')[0]
        tbody.innerHTML = '' // Clear existing rows
        if (this.activeTermsService === undefined) return
        for(const neighbourTerm of this.activeTermsService.getQueryTerm().getNeighbourTerms()) {
            const row = tbody.insertRow()
            const cell1 = row.insertCell(0)
            const cell2 = row.insertCell(1)

            cell1.innerHTML = neighbourTerm.getValue()
            cell2.innerHTML = neighbourTerm.getHops().toFixed(1)
        }
    }
}

class QueryTermService {
    private queryService: QueryService
    private queryTerm: QueryTerm
    private isVisible: boolean = false

    constructor(queryService: QueryService, queryTerm: QueryTerm) {
        this.queryService = queryService
        this.queryTerm = queryTerm
        this.retrieveData();
    }

    public getQueryTerm(): QueryTerm {
        return this.queryTerm
    }

    public nodeDragged(id: string, position: Position): void {
        const term: NeighbourTerm | undefined = this.queryTerm.getNeighbourTermById(id)
        if (term === undefined) return
        term.setPosition(position)

        this.queryService.updateNeighbourTermsTable()
    }

    public display(): void {
        this.isVisible = true
        this.queryTerm.removeViews()
        this.queryTerm.displayViews()
        this.center()
    }

    public deactivate(): void {
        this.isVisible = false
        this.queryTerm.removeViews()
    }

    public removeNeighbourTerm(id: string): void {
        const neighbourTerm = this.queryTerm.getNeighbourTermById(id)
        if (neighbourTerm === undefined) return
        this.queryTerm.removeNeighbourTerm(neighbourTerm)
        this.queryService.updateNeighbourTermsTable()
    }

    private center(): void {
        cy.zoom(1.2)
        if (this.queryTerm.getNode() === undefined) return 
        cy.center(cy.getElementById((this.queryTerm.getNode() as CentralNode).getId()))
    }

    private async retrieveData() {
        const endpoint = 'get-neighbour-terms'
        const result = await ApiUtils.postData(endpoint, { query: this.queryTerm.getValue() })
        if (result) {
            for (let termObject of result['neighbour_terms']) {
                const neighbourTerm = new NeighbourTerm(this.queryTerm)
                neighbourTerm.setLabel(termObject.term)
                neighbourTerm.setHops(termObject.distance)
                this.addNeighbourTerm(neighbourTerm)
            }
        }
    }

    private addNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        this.queryTerm.addNeighbourTerm(neighbourTerm)

        this.queryService.updateNeighbourTermsTable()
        if (this.isVisible) this.display()
    }
}


class Query {
    private query: string = ''
    private queryTerms: QueryTerm[] = []

    constructor(query: string) {
        this.query = query
    }

    public setQuery(query: string): void {
        this.query = query
    }

    public getQuery(): string {
        return this.query
    }

    public setQueryTerms(queryTerms: QueryTerm[]): void {
        this.queryTerms = queryTerms
    }

    public getQueryTerms(): QueryTerm[] {
        return this.queryTerms
    }
}


class QueryTermsList {
    private dynamicList: HTMLElement
    private queryService: QueryService

    constructor(queryService: QueryService) {
        this.queryService = queryService
        this.dynamicList = document.getElementById('queryTermsList') as HTMLElement
    }

    public updateList(queryTerms: QueryTerm[]): void {
        this.dynamicList.innerHTML = ''
        queryTerms.forEach(queryTerm => {
            // Create a new list item element
            const listItem = document.createElement("li")
            listItem.textContent = queryTerm.getValue()

            listItem.addEventListener("click", () => {
                this.queryService.setActiveTermsService(queryTerm.getValue())
            })

            // Append the list item to the dynamic list container
            this.dynamicList.appendChild(listItem)
        })
    }
}


class QueryService {
    public activeQueryTermService: QueryTermService | undefined
    private queryTermServices: QueryTermService[] = []
    private neighbourTermsTable: NeighbourTermsTable
    private queryTermsList: QueryTermsList
    private query: Query

    constructor() {
        this.neighbourTermsTable = new NeighbourTermsTable()
        this.queryTermsList = new QueryTermsList(this)
        this.query = new Query('')
    }

    public setQuery(query: string): void {
        this.activeQueryTermService?.deactivate()
        
        this.query = new Query(query)
        this.queryGenerationWasRequested()

        if (this.queryTermServices.length === 0) return
        this.setActiveTermsService(this.queryTermServices[0].getQueryTerm().getValue())
    }

    public setActiveTermsService(queryTerm: string): void {
        this.activeQueryTermService?.deactivate()
        const queryTermService = this.findQueryTermService(queryTerm)
        if (queryTermService === undefined) return
        this.activeQueryTermService = queryTermService
        this.activeQueryTermService.display()
        this.neighbourTermsTable.setActiveService(queryTermService)
        this.updateNeighbourTermsTable()
        return
    }

    public updateNeighbourTermsTable(): void {
        this.neighbourTermsTable.updateTable()
    }

    private queryGenerationWasRequested(): void {
        const queryTermService = new QueryTermService(this, new QueryTerm(this.query.getQuery()))

        this.queryTermServices = []
        this.queryTermServices.push(queryTermService)
        // this.decomposeQuery()
        this.queryTermsList.updateList(
            this.queryTermServices.map(termService => termService.getQueryTerm())
        )
        if (this.queryTermServices.length > 0) {
            this.activeQueryTermService = this.queryTermServices[0]
            this.neighbourTermsTable.setActiveService(this.activeQueryTermService)
        }
    }

    private findQueryTermService(queryTermValue: string): QueryTermService | undefined {
        return this.queryTermServices.find(
            termService => termService.getQueryTerm().getValue() === queryTermValue
        )
    }

    private decomposeQuery(): void {
        this.queryTermServices = []
        for (let term of this.query.getQuery().split(' ')) {
            const termService = new QueryTermService(this, new QueryTerm(term))
            this.queryTermServices.push(termService)
        }
    }
}


class QueryComponent {
    private query: string = ''
    private queryService: QueryService
    private input: HTMLInputElement

    constructor(queryService: QueryService) {
        this.queryService = queryService
        this.input = document.getElementById('queryInput') as HTMLInputElement

        this.input.addEventListener("keyup", event => {
            if(event.key !== "Enter") return
            event.stopImmediatePropagation()
            this.query = this.input.value.trim()
            this.input.value = ''
            event.preventDefault()
            this.queryService.setQuery(this.query)
        })
    }
}





const cy = cytoscape({
    container: document.getElementById("cy") as HTMLElement,
    layout: {
        name: "preset",
    },
    style: [
        {
            selector: '.' + NodeType.central_node,
            style: {
            "background-color": 'red',
            'width': '20px',
            'height': '20px',
            'label': "data(id)",
            },
        },
        {
            selector: "edge",
            style: {
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "line-color": "#ccc",
            label: "data(distance)",
            "width": "2px", // set the width of the edge
            "font-size": "12px" // set the font size of the label            
            },
        },
        {
            selector: '.' + NodeType.outer_node,
            style: {
              'background-color': 'blue',
              'width': '15px',
              'height': '15px',
              'label': 'data(label)'
            }
        }
    ],
    userZoomingEnabled: false,
    userPanningEnabled: false
})


cy.on('drag', 'node', evt => {
    queryService.activeQueryTermService?.nodeDragged(evt.target.id(), evt.target.position())
})


const queryService: QueryService = new QueryService()
const queryComponent: QueryComponent = new QueryComponent(queryService)

cy.ready(() => {
    // queryService.queryTermServices[0].addNeighbourTerm('holaA')
    // queryService.queryTermServices[0].addNeighbourTerm('holaB')
    // queryService.queryTermServices[0].addNeighbourTerm('holaC')
    // queryService.queryTermServices[1].addNeighbourTerm('mundoA')
    // queryService.queryTermServices[1].addNeighbourTerm('mundoB')
    // queryService.queryTermServices[1].addNeighbourTerm('mundoC')
})

let searchTerm = "Graphs"

const mockResults = [
    "Paper 1 about " + searchTerm,
    "Paper 2 related to " + searchTerm,
    "Another paper discussing " + searchTerm,
    "Further findings on " + searchTerm
]

// Display results
const resultsList = document.getElementById('resultsList') as HTMLElement
resultsList.innerHTML = "" // Clear previous results

for (let i = 0; i < mockResults.length; i++) {
    let listItem = document.createElement('li')
    listItem.textContent = (i + 1) + ". " + mockResults[i]
    resultsList.appendChild(listItem)

    listItem.addEventListener("click", () => {
        alert(mockResults[i])
    })
}

// quick way to get instances in console
;(window as any).cy = cy
;(window as any).queryService = queryService
