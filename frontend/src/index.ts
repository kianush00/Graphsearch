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


class HopConversionUtils {
    // While max distance is always 200, then -> 200 / (user distance) = hopToDistanceRatio
    private static hopToDistanceRatio: number = 50;

    public static convertHopsToDistance(hops: number): number {
        return hops * this.hopToDistanceRatio
    }

    public static convertDistanceToHops(distance: number): number {
        return parseFloat((distance / this.hopToDistanceRatio).toFixed(1))
    }
}


class HTTPRequestUtils {
    /**
     * Sends a POST request to the specified endpoint with the provided data.
     *
     * @param endpoint - The endpoint to which the request will be sent.
     * @param data - The data to be sent in the request body.
     *
     * @returns A Promise that resolves with the response data.
     */
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
        this.distance = distance
        const cyEdge = cy.edges(`[source = "${this.sourceNode.getId()}"][target = "${this.targetNode.getId()}"]`)
        cyEdge.data('distance', HopConversionUtils.convertDistanceToHops(this.distance))
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
                distance: HopConversionUtils.convertDistanceToHops(this.distance)
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
        this.id = id
        this.label = label
        this.position = position
        this.type = type
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
        // Generates a random angle from the provided distance, to calculate the new position
        let _position = MathUtils.getAngularPosition(MathUtils.getRandomAngle(), distance)
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

    private getDistance(): number {
        return MathUtils.calculateEuclideanDistance(this.position.x, this.position.y)
    }

    private updateVisualPosition(): void {
        cy.getElementById(this.id).position(this.position)
    }

    private addVisualNodeToInterface(): void {
        cy.add(this.toObject()).addClass(this.type.toString())
    }
}


/**
 * Represents a term in a graph.
 * It is associated with a graph node.
 */
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


interface ViewManager {
    displayViews(): void
    removeViews(): void
}


/**
 * Represents a neighbour term in the graph.
 * It manages the associated views, such as the OuterNode and Edge.
 */
class NeighbourTerm extends Term implements ViewManager {
    protected node: OuterNode | undefined
    private queryTerm: QueryTerm
    private hops: number = 0
    private nodePosition: Position = { x: 0, y: 0 }
    private edge: Edge | undefined

    constructor(queryTerm: QueryTerm, value: string, hops: number) {
        super(value)
        this.queryTerm = queryTerm
        this.setLabel(value)
        this.initializeHopsAndNodePosition(hops)
    }

    /**
     * Displays the views of the neighbour term in the graph.
     * This includes creating and positioning the OuterNode and Edge.
     */
    public displayViews(): void {
        this.node = new OuterNode(TextUtils.getRandomString(24))
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

    /**
     * Sets the position of the neighbour term node and updates the neighbour term's hops.
     *
     * @param position - The new position of the neighbour term node.
     *
     * @returns {void}
     */
    public setPosition(position: Position): void {
        const nodeDistance = this.edge?.getDistance() ?? 0
        this.nodePosition = this.validatePositionWithinRange(position, nodeDistance)
        const distance = MathUtils.calculateEuclideanDistance(this.nodePosition.x, this.nodePosition.y)
        this.hops = HopConversionUtils.convertDistanceToHops(distance)
        this.updateNodePosition()
    }

    /**
     * Sets the number of hops for the neighbour term node and updates the neighbour term's position.
     *
     * @param hops - The new number of hops for the neighbour term node.
     *
     * @returns {void}
     */
    private initializeHopsAndNodePosition(hops: number): void {
        this.hops = hops
        const nodeDistance = HopConversionUtils.convertHopsToDistance(hops)
        this.nodePosition = MathUtils.getRandomAngularPositionWithDistance(nodeDistance)
        this.updateNodePosition()
    }

    private validatePositionWithinRange(position: Position, nodeDistance: number): Position {
        let positionDistance = MathUtils.calculateEuclideanDistance(position.x, position.y)

        if (this.edge !== undefined && this.node !== undefined ) {
            if (positionDistance < 50.0 || positionDistance > 200.0) {
                let angle = Math.atan2(position.y, position.x)
                let adjustedX = Math.cos(angle) * nodeDistance
                let adjustedY = Math.sin(angle) * nodeDistance
                position.x = adjustedX
                position.y = adjustedY
            }
        }
        return position
    }

    private updateNodePosition(): void {
        this.node?.setPosition(this.nodePosition)
        this.edge?.updateDistance()
    }
}


/**
 * Represents a query term that is associated with a central node in the graph.
 * It also manages neighbour terms related to the query term.
 */
class QueryTerm extends Term implements ViewManager {
    protected node: CentralNode | undefined
    private neighbourTerms: NeighbourTerm[] = []

    constructor(value: string) {
        super(value)
    }

    /**
     * Displays the views of the query term and its associated neighbour terms in the graph.
     * This includes creating and positioning the CentralNode and OuterNodes.
     */
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


class Document {
    private queryTerm: QueryTerm
    private id: string
    private title: string
    private abstract: string

    constructor(queryTermValue: string, id: string, title: string, abstract: string) {
        this.queryTerm = new QueryTerm(queryTermValue)
        this.id = id
        this.title = title
        this.abstract = abstract
    }

    public getQueryTerm(): QueryTerm {
        return this.queryTerm
    }

    public getId(): string {
        return this.id
    }

    public getTitle(): string {
        return this.title
    }

    public getAbstract(): string {
        return this.abstract
    }
    
}


class Ranking {
    private visibleQueryTerm: QueryTerm
    private completeQueryTerm: QueryTerm
    private documents: Document[] = []

    constructor(queryTermValue: string) {
        this.visibleQueryTerm = new QueryTerm(queryTermValue)
        this.completeQueryTerm = new QueryTerm(queryTermValue)
    }

    public getVisibleQueryTerm(): QueryTerm {
        return this.visibleQueryTerm
    }

    public getCompleteQueryTerm(): QueryTerm {
        return this.completeQueryTerm
    }

    public getDocuments(): Document[] {
        return this.documents
    }

    public addDocument(document: Document): void {
        this.documents.push(document)
    }
}


class QueryTermService {
    private queryService: QueryService
    private ranking: Ranking
    private isVisible: boolean = false

    constructor(queryService: QueryService, queryTermValue: string) {
        this.queryService = queryService
        this.ranking = new Ranking(queryTermValue)
        this.retrieveData()
    }

    public getVisibleQueryTerm(): QueryTerm {
        return this.ranking.getVisibleQueryTerm()
    }

    public getRanking(): Ranking {
        return this.ranking
    }

    /**
     * If the node is dragged, updates the position of the neighbour term node and 
     * updates the neighbour term's hops.
     * @param id - The id of the neighbour term node.
     * @param position - The new position of the neighbour term node.
     */
    public nodeDragged(id: string, position: Position): void {
        const term: NeighbourTerm | undefined = this.getVisibleQueryTerm().getNeighbourTermById(id)
        if (term === undefined) return
        term.setPosition(position)

        // Update the neighbour terms table with the new hops values
        this.queryService.updateNeighbourTermsTable()
    }

    /**
     * Displays the QueryTerm and its associated views in the graph.
     * This includes creating and positioning the CentralNode and OuterNodes.
     */
    public display(): void {
        this.isVisible = true // Mark the QueryTerm as visible

        // Remove any existing views associated with the QueryTerm
        this.getVisibleQueryTerm().removeViews()

        // Display the views associated with the QueryTerm
        this.getVisibleQueryTerm().displayViews()

        // Center the graph on the CentralNode
        this.center()
    }

    /**
     * This method removes the visual nodes and edges from the graph interface.
     */
    public deactivate(): void {
        this.isVisible = false
        this.getVisibleQueryTerm().removeViews()
    }

    public removeNeighbourTerm(id: string): void {
        const neighbourTerm = this.getVisibleQueryTerm().getNeighbourTermById(id)
        if (neighbourTerm === undefined) return
        this.getVisibleQueryTerm().removeNeighbourTerm(neighbourTerm)
        this.queryService.updateNeighbourTermsTable()
    }

    private center(): void {
        cy.zoom(1.2)
        if (this.getVisibleQueryTerm().getNode() === undefined) return 
        cy.center(cy.getElementById((this.getVisibleQueryTerm().getNode() as CentralNode).getId()))
    }

    /**
     * Retrieves data related to neighbour terms for the current query term.
     * The retrieved data is then used to create new NeighbourTerm instances,
     * which are added to the QueryTerm's neighbour terms list.
     *
     * @returns {Promise<void>} - A promise that resolves when the data retrieval and processing are complete.
     */
    private async retrieveData() {
        // Define the endpoint for retrieving neighbour terms data
        const endpoint = 'get-ranking'

        // Send a POST request to the endpoint with the query term value
        let _query = this.getVisibleQueryTerm().getValue()
        const result = await HTTPRequestUtils.postData(endpoint, { query: _query })

        // Check if the result is not null
        if (result) {
            this.generateVisibleNeighbourTerms(result)
            this.generateRankingDocuments(result)
        }
    }

    private generateVisibleNeighbourTerms(result: any) {
        // Iterate over the neighbour terms in the result
        for (let termObject of result['visible_neighbour_terms']) {
            // Create a new NeighbourTerm instance for each term object
            const neighbourTerm = new NeighbourTerm(this.getVisibleQueryTerm(), termObject.term, termObject.distance)

            // Add the neighbour term to the QueryTerm's neighbour terms list
            this.addNeighbourTerm(neighbourTerm)
        }
    }

    private generateRankingDocuments(result: any) {
        // Iterate over the documents in the result
        for (let documentObject of result['documents']) {
            const id = documentObject['doc_id']
            const title = documentObject['title']
            const abstract = documentObject['abstract']
            const document = new Document(this.ranking.getVisibleQueryTerm().getValue(), id, title, abstract)
            this.addDocument(document)
        }
    }

    /**
     * Adds a neighbour term to the QueryTerm's neighbour terms list.
     * It also updates the neighbour terms table in the QueryService.
     * If the QueryTerm is currently visible, it displays the views of the neighbour term.
     *
     * @param neighbourTerm - The neighbour term to be added.
     * @returns {void}
     */
    private addNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        this.getVisibleQueryTerm().addNeighbourTerm(neighbourTerm)
        this.queryService.updateNeighbourTermsTable()
        if (this.isVisible) this.display()
    }

    private addDocument(document: Document): void {
        this.getRanking().addDocument(document)
        this.queryService.updateResultsList()
    }
}


class NeighbourTermsTable {
    private activeTermService: QueryTermService | undefined
    private dynamicTable: HTMLElement

    constructor() {
        this.dynamicTable = document.getElementById('neighboursTermsTable') as HTMLElement
    }

    public setActiveTermService(queryTermService: QueryTermService): void {
        this.activeTermService = queryTermService
    }

    /**
     * Updates the table with the values of neighbour terms.
     * Clears existing rows in the table before adding new ones.
     */
    public updateTable(): void {
        // Get the table body element
        const tbody = this.dynamicTable.getElementsByTagName('tbody')[0]

        // Clear existing rows in the table
        tbody.innerHTML = '' 

        // Check if the activeTermService is defined
        if (this.activeTermService === undefined) return

        // Iterate over the neighbour terms of the active query term
        for(const neighbourTerm of this.activeTermService.getVisibleQueryTerm().getNeighbourTerms()) {
            // Create a new row in the table
            const row = tbody.insertRow()

            // Create cells for the row
            const cell1 = row.insertCell(0)
            const cell2 = row.insertCell(1)

            // Set the text content of the cells
            cell1.innerHTML = neighbourTerm.getValue()
            cell2.innerHTML = neighbourTerm.getHops().toFixed(1)
        }
    }
}


class ResultsList {
    private activeTermService: QueryTermService | undefined
    private dynamicList: HTMLElement

    constructor() {
        this.dynamicList = document.getElementById('resultsList') as HTMLElement
    }

    public setActiveTermService(queryTermService: QueryTermService): void {
        this.activeTermService = queryTermService
    }

    public updateList(): void {
        // Clear existing list items
        this.dynamicList.innerHTML = '';

        // Check if the activeTermService is defined
        if (this.activeTermService === undefined) return

        // Get the ranking of the active query term
        let documents = this.activeTermService.getRanking().getDocuments()
    
        for (let i = 0; i < documents.length; i++) {
            // Create a new list item element
            const listItem = document.createElement('li');
    
            // Create a title element
            const titleElement = document.createElement('span');
            titleElement.textContent = (i + 1) + ". " + documents[i].getTitle();
    
            // Create an abstract element
            const abstractElement = document.createElement('p');
            abstractElement.textContent = documents[i].getAbstract();
    
            // Add a click event listener to the list item
            listItem.addEventListener('click', () => {
                // When the list item is clicked, opens the original URL document webpage in a new tab
                window.open('https://ieeexplore.ieee.org/document/' + documents[i].getId(), '_blank');
            });
    
            // Append the title and abstract to the list item
            listItem.appendChild(titleElement);
            listItem.appendChild(abstractElement);
    
            // Append the list item to the dynamic list container
            this.dynamicList.appendChild(listItem);
        }
    }
    
}


class QueryTermsList {
    private dynamicList: HTMLElement
    private queryService: QueryService

    constructor(queryService: QueryService) {
        this.queryService = queryService
        this.dynamicList = document.getElementById('queryTermsList') as HTMLElement
    }

    /**
     * Updates the list of query terms with new query terms.
     *
     * @param queryTerms - An array of QueryTerm objects to be displayed in the list.
     */
    public updateList(queryTerms: QueryTerm[]): void {
        // Clear existing list items
        this.dynamicList.innerHTML = ''

        // Iterate over the query terms and create list items for each one
        queryTerms.forEach(queryTerm => {
            // Create a new list item element
            const listItem = document.createElement("li")

            // Set the text content of the list item to be the value of the query term
            listItem.textContent = queryTerm.getValue()

            // Add a click event listener to the list item
            listItem.addEventListener("click", () => {
                // When the list item is clicked, set the active query term service to the value of the query term
                this.queryService.setActiveQueryTermService(queryTerm.getValue())
            })

            // Append the list item to the dynamic list container
            this.dynamicList.appendChild(listItem)
        })
    }
}


class QueryService {
    private activeQueryTermService: QueryTermService | undefined
    private queryTermServices: QueryTermService[]
    private neighbourTermsTable: NeighbourTermsTable
    private queryTermsList: QueryTermsList
    private resultsList: ResultsList

    constructor() {
        this.queryTermServices = []
        this.neighbourTermsTable = new NeighbourTermsTable()
        this.resultsList = new ResultsList()
        this.queryTermsList = new QueryTermsList(this)
        new QueryComponent(this)
    }

    /**
     * Sets the query for the service.
     * Deactivates the currently active QueryTermService, creates a new Query object,
     * and triggers the query generation process.
     * @param query - The new query string.
     */
    public setQuery(queryValue: string): void {
        this.activeQueryTermService?.deactivate()
        this.generateNewQueryTermService(queryValue)
        if (this.queryTermServices.length > 0) {
            this.setActiveQueryTermService(queryValue)
        }
    }

    public getActiveQueryTermService(): QueryTermService | undefined { 
        return this.activeQueryTermService 
    }

    /**
     * Sets the active QueryTermService based on the provided query value.
     * Deactivates the currently active QueryTermService, finds the corresponding QueryTermService,
     * by the provided queryValue, and displays the views associated with the QueryTerm.
     *
     * @param queryValue - The value of the query term for which to set the active QueryTermService.
     */
    public setActiveQueryTermService(queryValue: string): void {
        this.activeQueryTermService?.deactivate()
        const queryTermService = this.findQueryTermService(queryValue)
        if (queryTermService !== undefined) {
            this.activeQueryTermService = queryTermService
            this.activeQueryTermService.display()

            this.neighbourTermsTable.setActiveTermService(this.activeQueryTermService)
            this.resultsList.setActiveTermService(this.activeQueryTermService)
            this.updateNeighbourTermsTable()
            this.updateResultsList()
        }
    }

    public updateNeighbourTermsTable(): void {
        this.neighbourTermsTable.updateTable()
    }

    public updateResultsList(): void {
        this.resultsList.updateList()
    }

    /**
     * Generates a new QueryTermService for a given query value.
     * This method checks if a QueryTermService for the given query value already exists.
     * If not, it creates a new QueryTermService, adds it to the queryTermServices array,
     * and updates the query terms list.
     *
     * @param queryValue - The value of the query term for which to generate a new QueryTermService.
     */
    private generateNewQueryTermService(queryValue: string): void {
        if (this.findQueryTermService(queryValue) === undefined) {
            const queryTermService = new QueryTermService(this, queryValue)
            this.queryTermServices.push(queryTermService)
            this.updateQueryTermsList()
        }
    }

    private findQueryTermService(queryValue: string): QueryTermService | undefined {
        return this.queryTermServices.find(
            termService => termService.getVisibleQueryTerm().getValue() === queryValue
        )
    }

    private updateQueryTermsList(): void {
        this.queryTermsList.updateList(
            this.queryTermServices.map(termService => termService.getVisibleQueryTerm())
        )
    }
}


/**
 * Represents a component responsible for handling query input interactions.
 * 
 * This class is responsible for capturing user input from an HTML input element,
 * and sending the query to a query service when the Enter key is pressed.
 */
class QueryComponent {
    private queryService: QueryService
    private input: HTMLInputElement

    constructor(queryService: QueryService) {
        this.queryService = queryService
        this.input = document.getElementById('queryInput') as HTMLInputElement

        // Add event listener to the input element to handle "Enter" key presses
        this.input.addEventListener("keyup", event => {
            if(event.key !== "Enter") return // Only proceed if the "Enter" key is pressed
            event.stopImmediatePropagation() // Prevent other handlers from being called
            let queryValue = this.input.value.trim() // Get the trimmed input value
            this.input.value = '' // Clear the input field
            event.preventDefault() // Prevent the default action
            const alphanumericRegex = /[a-zA-Z0-9]/
            if (alphanumericRegex.test(queryValue)) {   // Check if the value contains at least one alphanumeric character
                this.queryService.setQuery(queryValue) // Send the query to the query service
            } else if (queryValue !== '') {
                alert("Please enter a valid query.")    // Alert the user if the query is invalid
            }
        })
    }
}


class RerankComponent {
    private queryService: QueryService
    private button: HTMLButtonElement

    constructor(queryService: QueryService) {
        this.queryService = queryService
        this.button = document.getElementById('rerankButton') as HTMLButtonElement

        // Add event listener to the button element
        this.button.addEventListener('click', this.handleRerankClick.bind(this))
    }

    private async handleRerankClick() {
        // Create the data to be sent in the POST request
        const data = {
            // Add the necessary data structure here
            message: "Rerank request"
        }

        // Send the POST request
        const response = await HTTPRequestUtils.postData('rerank', data)
        
        if (response) {
            // Handle the response accordingly

        }
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
    queryService.getActiveQueryTermService()?.nodeDragged(evt.target.id(), evt.target.position())
})

const queryService: QueryService = new QueryService()

cy.ready(() => {
    // queryService.queryTermServices[0].addNeighbourTerm('holaA')
    // queryService.queryTermServices[0].addNeighbourTerm('holaB')
    // queryService.queryTermServices[0].addNeighbourTerm('holaC')
    // queryService.queryTermServices[1].addNeighbourTerm('mundoA')
    // queryService.queryTermServices[1].addNeighbourTerm('mundoB')
    // queryService.queryTermServices[1].addNeighbourTerm('mundoC')
})


// quick way to get instances in console
;(window as any).cy = cy
;(window as any).queryService = queryService
