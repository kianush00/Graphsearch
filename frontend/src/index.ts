import cytoscape from "cytoscape";

class MathUtils {
    /**
     * Gets the distance between two nodes in a graph.
     *
     * @param node1 - The first node in the graph.
     * @param node2 - The second node in the graph.
     *
     * @returns The distance between the two nodes, rounded to the nearest whole number.
     */
    public static getDistanceBetweenNodes(node1: GraphNode, node2: GraphNode): number {
        const pos1 = node1.getPosition()
        const pos2 = node2.getPosition()

        const dx = pos1.x - pos2.x
        const dy = pos1.y - pos2.y

        return Math.round(this.calculateEuclideanDistance(dx, dy))
    }


    /**
     * Calculates the Euclidean distance between two points in a 2D space.
     *
     * @param dx - The difference in the x-coordinates of the two points.
     * @param dy - The difference in the y-coordinates of the two points.
     *
     * @returns The Euclidean distance between the two points.
     */
    public static calculateEuclideanDistance(dx: number, dy: number): number {
        return Math.sqrt(dx * dx + dy * dy);
    }


    /**
     * A static method that generates a random position within a circular area.
     * The position is represented as an object with x and y coordinates.
     *
     * @returns {Position} - An object representing the random position within the circular area.
     * The object has properties x and y, representing the coordinates of the position.
     *
     * @example
     * const randomPosition = MathUtils.getRandomAngularPosition();
     * console.log(randomPosition); // Output: { x: 123.45, y: 67.89 }
     */
    public static getRandomAngularPosition(): Position {
        const randomDistance = Math.random() * 200
        const randomAngle = Math.random() * 2 * Math.PI
        return this.getAngularPosition(randomAngle, randomDistance)
    }

    /**
     * A static method that generates a random angle in radians.
     *
     * @returns {number} A random angle between 0 (inclusive) and 2pi (exclusive).
     */
    public static getRandomAngle(): number {
        return Math.random() * 2 * Math.PI
    }

    /**
     * A static method that calculates the position of a node in a circular layout based on an angle and distance.
     *
     * @param angle - The angle in radians from the center of the circle.
     * @param distance - The distance from the center of the circle.
     *
     * @returns {Position} - An object containing the x and y coordinates of the node's position.
     *
     * @example
     * const angle = Math.PI / 4;
     * const distance = 100;
     * const position = QueryTermService.getAngularPosition(angle, distance);
     * console.log(position); // Output: { x: 70.71, y: 70.71 }
     */
    public static getAngularPosition(angle: number, distance: number): Position {
        return {
            x: distance * Math.cos(angle),
            y: distance * Math.sin(angle),
        }
    }

    /**
     * A static method that generates a random angular position with a given distance.
     *
     * @param distance - The distance from the origin for the position.
     * @returns {Position} - An object representing the position with x and y coordinates.
     *
     * @example
     * const randomPosition = MathUtils.getRandomAngularPositionWithDistance(100);
     * console.log(randomPosition); // Output: { x: 70.71, y: 70.71 }
     */
    public static getRandomAngularPositionWithDistance(distance: number): Position {
        const randomAngle = Math.random() * 2 * Math.PI
        return this.getAngularPosition(randomAngle, distance)
    }
}


class TextUtils {
    /**
     * A static utility function to generate a random string of a specified length.
     *
     * @param chars - The length of the random string to be generated.
     * @returns {string} - A random string of the specified length.
     *
     * @example
     * ```typescript
     * const randomString = getRandomString(10);
     * console.log(randomString); // Output: "aRFd4fK2Qj"
     * ```
     */
    public static getRandomString(chars: number): string {
        const charsList = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        let result = ''
        for (let i = 0; i < chars; i++) {
            result += charsList.charAt(Math.floor(Math.random() * charsList.length))
        }
        return result
    }

    /**
     * Separates a boolean query into individual terms.
     *
     * @param query - The boolean query to be separated.
     * @returns An array of strings representing the separated terms.
     *
     * @remarks
     * This function uses a regular expression to match boolean operators, parentheses, colons, and terms.
     * It then filters the matches, removing specified elements such as parentheses and boolean operators.
     * If no matches are found, an empty array is returned.
     */
    public static separateBooleanQuery(query: string): string[] {
        // Defines a regular expression to match boolean operators, parentheses, colons, and terms
        const pattern = /\bAND\b|\bOR\b|\bNOT\b|\(|\)|\w+|:/g;
        
        // Finds all matches using the regular expression
        let tokens = query.match(pattern);

        // Defines a set of elements to remove
        const elementsToRemove = new Set(['(', ')', 'AND', 'OR', 'NOT']);

        // Filters the array, removing the specified elements
        if (tokens !== null) {
            return tokens.filter(item => !elementsToRemove.has(item));
        }

        return []
    }
}


class HopConversionUtils {
    // While max distance is always 200, then -> 200 / (user distance) = hopToDistanceRatio
    private static hopToDistanceRatio: number = 50;

    /**
     * Converts the given number of hops to a distance in the graph.
     *
     * @param hops - The number of hops to convert.
     * @returns The distance in the graph corresponding to the given number of hops.
     *
     * @remarks
     * The conversion is based on a predefined ratio (hopToDistanceRatio).
     * The distance is calculated by multiplying the number of hops by the hopToDistanceRatio.
     */
    public static convertHopsToDistance(hops: number): number {
        return hops * this.hopToDistanceRatio
    }

    /**
     * Converts the given distance in the graph to a number of hops.
     *
     * @param distance - The distance in the graph to convert.
     * @returns The number of hops corresponding to the given distance.
     *
     * @remarks
     * The conversion is based on a predefined ratio (hopToDistanceRatio).
     * The number of hops is calculated by dividing the distance by the hopToDistanceRatio.
     * The result is rounded to one decimal place.
     */
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

    /**
     * Represents an edge in the graph, connecting two nodes.
     * 
     * @param sourceNode - The source node of the edge.
     * @param targetNode - The target node of the edge.
     */
    constructor(sourceNode: GraphNode, targetNode: GraphNode) {
        this.id = "e_" + targetNode.getId()
        this.sourceNode = sourceNode
        this.targetNode = targetNode
        this.distance = MathUtils.getDistanceBetweenNodes(sourceNode, targetNode)
        cy.add(this.toObject())
    }

    /**
     * Sets the distance for the edge between the source and target nodes.
     * This method updates the distance value and the corresponding edge data in the graph.
     *
     * @remarks
     * This method assumes that the source and target nodes are already associated with the edge.
     * It also assumes that the graph is represented using the Cytoscape.js library.
     */
    public setDistance(distance: number): void {
        this.distance = distance
        const cyEdge = cy.edges(`[source = "${this.sourceNode.getId()}"][target = "${this.targetNode.getId()}"]`)
        cyEdge.data('distance', HopConversionUtils.convertDistanceToHops(this.distance))
    }

    /**
     * Updates the distance between the source and target nodes.
     *
     * This function calculates the distance between the source and target nodes using the `MathUtils.getDistanceBetweenNodes` method.
     * It then sets the calculated distance as the distance between the nodes using the `this.setDistance` method.
     *
     * @remarks
     * This function assumes that the source and target nodes are already defined and valid.
     */
    public updateDistance(): void {
        this.setDistance(MathUtils.getDistanceBetweenNodes(this.sourceNode, this.targetNode))
    }

    public getDistance(): number {
        return this.distance
    }

    /**
     * Removes the visual node from the graph interface.
     * 
     * This function removes the node with the given ID from the graph interface.
     * It uses the Cytoscape.js library to select the node by its ID and remove it from the graph.
     *
     * @remarks
     * This function should be called when the node is no longer needed in the graph interface.
     * It ensures that the node is removed from the visual representation of the graph.
     */
    public remove(): void {
        cy.remove(cy.getElementById(this.id))
    }

    /**
     * Converts the Edge instance into a serializable object.
     * 
     * @returns An object containing the data of the Edge.
     * The object has properties: id, source, target, and distance.
     * The distance is converted to hops using the HopConversionUtils.
     */
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
    
    /**
     * Represents a node in the graph.
     * 
     * @param id - The unique identifier of the node.
     * @param label - The label of the node.
     * @param position - The position of the node in the graph.
     * @param type - The type of the node.
     */
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

    /**
     * Sets the label of the node and updates the associated graph node.
     *
     * @param label - The new label for the node.
     *
     * @remarks
     * This function updates the label of the node and also updates the label in the graph interface.
     * It uses the Cytoscape.js library to select the node by its ID and update the label data.
     */
    public setLabel(label: string): void {
        this.label = label
        cy.getElementById(this.id).data('label', label)
    }

    /**
     * Removes the visual node from the graph interface.
     *
     * @remarks
     * This function removes the node with the given ID from the graph interface.
     * It uses the Cytoscape.js library to select the node by its ID and remove it from the graph.
     */
    public remove(): void {
        cy.remove(cy.getElementById(this.id))
    }

    /**
     * Converts the GraphNode instance into a serializable object.
     *
     * @returns An object containing the data of the GraphNode.
     * The object has properties: id, label, and position.
     *
     * @remarks
     * This function is responsible for converting the GraphNode instance into a serializable object.
     * It returns an object containing the node's data and position.
     */
    public toObject(): { data: NodeData; position: Position } {
        return {
            data: {
                id: this.getId(),
                label: this.label,
            },
            position: this.position,
        }
    }
}


class CentralNode extends GraphNode {
    /**
     * Represents a central node in the graph.
     * It manages the associated views, such as the visual node in the graph interface.
     * 
     * @param id - The unique identifier of the central node.
     * @param x - The x-coordinate of the central node's position in the graph.
     * @param y - The y-coordinate of the central node's position in the graph.
     */
    constructor(id: string, x: number, y: number) {
        let _id = id
        let _label = id
        let _position = { x, y }
        let _type = NodeType.central_node
        super(_id, _label, _position, _type)
        this.addVisualNodeToInterface()
    }

    /**
     * Adds a visual node to the graph interface.
     * 
     * This function creates a new node in the graph using the `toObject` method,
     * which returns an object containing the node's data and position.
     * The node is then added to the graph using the `cy.add` method.
     * The node's class is set to the string representation of the node's type.
     * The node is also locked and ungrabified to prevent user interaction.
     */
    private addVisualNodeToInterface(): void {
        cy.add(this.toObject()).addClass(this.type.toString()).lock().ungrabify()
    }
}


class OuterNode extends GraphNode {
    /**
     * Represents an outer node in the graph.
     * It manages the associated views, such as the visual node in the graph interface.
     * 
     * @param id - The unique identifier of the outer node.
     * @param distance - The distance from the central node to the outer node.
     *                    Default value is 0, which means the outer node will be positioned randomly.
     */
    constructor(id: string, distance: number = 0) {
        let _id = id
        let _label = id
        // Generates a random angle from the provided distance, to calculate the new position
        let _position = MathUtils.getAngularPosition(MathUtils.getRandomAngle(), distance)
        let _type = NodeType.outer_node
        super(_id, _label, _position, _type)
        this.addVisualNodeToInterface()
    }

    /**
     * Sets the position of the OuterNode in the graph.
     *
     * @param position - The new position for the OuterNode.
     *
     * @remarks
     * This function updates the position of the OuterNode and also updates the position in the graph interface.
     */
    public setPosition(position: Position): void {
        this.position = position;
        this.updateVisualPosition();
    }

    /**
     * Sets the position of the OuterNode in the graph using an angle.
     *
     * @param angle - The angle from which to calculate the new position for the OuterNode.
     *
     * @remarks
     * This function calculates the new position of the OuterNode using the provided angle and the current distance from the central node.
     * It then updates the position of the OuterNode and the position in the graph interface.
     */
    public setPositionFromAngle(angle: number): void {
        this.position = MathUtils.getAngularPosition(angle, this.getDistance());
        this.updateVisualPosition();
    }

    /**
     * Sets the position of the OuterNode in the graph using a distance.
     *
     * @param distance - The distance from the central node to calculate the new position for the OuterNode.
     *
     * @remarks
     * This function calculates the new position of the OuterNode using a random angle and the provided distance.
     * It then updates the position of the OuterNode and the position in the graph interface.
     */
    public setPositionFromDistance(distance: number): void {
        this.position = MathUtils.getAngularPosition(MathUtils.getRandomAngle(), distance);
        this.updateVisualPosition();
    }

    /**
     * Calculates the distance from the central node to the OuterNode.
     *
     * @returns {number} - The distance from the central node to the OuterNode.
     *
     * @remarks
     * This function calculates the distance from the central node to the OuterNode using the Euclidean distance formula.
     */
    private getDistance(): number {
        return MathUtils.calculateEuclideanDistance(this.position.x, this.position.y);
    }

    /**
     * Updates the visual position of the OuterNode in the graph interface.
     *
     * @remarks
     * This function is responsible for updating the position of the OuterNode in the graph interface.
     * It sets the position of the OuterNode to the current position of the OuterNode instance.
     */
    private updateVisualPosition(): void {
        cy.getElementById(this.id).position(this.position)
    }

    /**
     * Adds a visual node to the graph interface.
     * 
     * This function creates a new node in the graph using the `toObject` method,
     * which returns an object containing the node's data and position.
     * The node is then added to the graph using the `cy.add` method.
     * The node's class is set to the string representation of the node's type.
     */
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

    /**
     * Represents a term in a graph.
     * It is associated with a graph node.
     */
    constructor(value: string) {
        this.value = value
    }

    /**
     * Sets the label of the term and updates the associated graph node.
     * 
     * @param value - The new label for the term.
     */
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

interface NTermObject {
    term: string;
    ponderation: number;
    distance: number;
}


class NeighbourTerm extends Term implements ViewManager {
    protected node: OuterNode | undefined
    private queryTerm: QueryTerm
    private hops: number = 0
    private nodePosition: Position = { x: 0, y: 0 }
    private edge: Edge | undefined
    private ponderation: number

    /**
     * Represents a neighbour term in the graph.
     * It manages the associated views, such as the OuterNode and Edge.
     * 
     * @param queryTerm - The QueryTerm associated with this neighbour term.
     * @param value - The value of the neighbour term.
     * @param hops - The number of hops from the central node to this neighbour term.
     * @param ponderation - The ponderation of this neighbour term.
     */
    constructor(queryTerm: QueryTerm, value: string, hops: number, ponderation: number) {
        super(value)
        this.queryTerm = queryTerm
        this.ponderation = ponderation
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

    /**
     * Removes the views of the neighbour terms and the central node.
     * 
     * This function is responsible for removing the visual nodes (OuterNodes and CentralNode)
     * and edges (connecting the CentralNode to the OuterNodes) from the graph interface.
     */
    public removeViews(): void {
        this.node?.remove()
        this.edge?.remove()
    }

    public getHops(): number {
        return this.hops
    }

    public getPonderation(): number {
        return this.ponderation
    }

    /**
     * Converts the NeighbourTerm instance into an object containing term, ponderation, and distance.
     * 
     * @returns {NTermObject} - An object with properties term, ponderation, and distance.
     * The term property contains the value of the NeighbourTerm instance.
     * The ponderation property contains the ponderation of the NeighbourTerm instance.
     * The distance property contains the number of hops from the central node to the NeighbourTerm instance.
     */
    public toObject(): NTermObject {
        return {
            term: this.value,
            ponderation: this.ponderation,
            distance: this.hops
        }
    }

    /**
     * Sets the position of the neighbour term node and updates the neighbour term's hops.
     *
     * @param position - The new position of the neighbour term node.
     * The position object contains properties x and y representing the coordinates of the new position.
     *
     * @remarks
     * This function calculates the distance between the new position and the central node,
     * validates the position to ensure it falls within the specified range, updates the number of hops,
     * and updates the position of the neighbour term node.
     *
     * @returns {void} - This function does not return any value.
     */
    public setPosition(position: Position): void {
        const nodeDistance = this.edge?.getDistance() ?? 0
        this.nodePosition = this.validatePositionWithinRange(position, nodeDistance)
        const distance = MathUtils.calculateEuclideanDistance(this.nodePosition.x, this.nodePosition.y)
        this.hops = HopConversionUtils.convertDistanceToHops(distance)
        this.updateNodePosition()
    }

    /**
     * Initializes the number of hops and the position of the neighbour term node.
     *
     * @param hops - The number of hops for the neighbour term node.
     * This value is used to calculate the distance from the central node.
     *
     * @remarks
     * This function sets the number of hops for the neighbour term node,
     * calculates the distance from the central node based on the number of hops,
     * generates a random position for the neighbour term node within the specified range,
     * and updates the position of the neighbour term node.
     *
     * @returns {void} - This function does not return any value.
     */
    private initializeHopsAndNodePosition(hops: number): void {
        this.hops = hops
        const nodeDistance = HopConversionUtils.convertHopsToDistance(hops)
        this.nodePosition = MathUtils.getRandomAngularPositionWithDistance(nodeDistance)
        this.updateNodePosition()
    }

    /**
     * Validates the position of a neighbour term node within a specified range.
     * If the position is outside the range, it adjusts the position to be within the range.
     *
     * @param position - The position of the neighbour term node to be validated.
     * @param nodeDistance - The distance of the neighbour term node from the central node.
     *
     * @returns {Position} - The validated position of the neighbour term node.
     *
     * @remarks
     * This function checks if the position of the neighbour term node is within the specified range (50.0 to 200.0 units from the central node).
     * If the position is outside the range, it adjusts the position to be within the range by calculating the adjusted X and Y coordinates.
     * The adjusted position is then returned.
     */
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

    /**
     * Updates the position of the neighbour term node and updates the neighbour term's hops.
     *
     * @remarks
     * This function is responsible for updating the position of the neighbour term node and
     * the number of hops for the neighbour term. It calls the `setPosition` method of the neighbour term node
     * and the `updateDistance` method of the edge connecting the neighbour term node to the central node.
     *
     * 
     */
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

    public setNeighbourTerms(neighbourTerms: NeighbourTerm[]): void {
        this.neighbourTerms = neighbourTerms
    }

    public getNeighbourTermsValues(): string[] {
        return this.neighbourTerms.map(term => term.getValue())
    }

    public getNeighbourTermsAsObjects(): NTermObject[] {
        return this.neighbourTerms.map(term => term.toObject())
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


interface DocumentObject {
    doc_id: string;
    title: string;
    abstract: string;
    neighbour_terms: NTermObject[];
}

class Document {
    private queryTerm: QueryTerm
    private id: string
    private title: string
    private abstract: string

    /**
     * Represents a document associated with a query term.
     * 
     * @param queryTermValue - The value of the query term associated with the document.
     * @param id - The unique identifier of the document.
     * @param title - The title of the document.
     * @param abstract - The abstract of the document.
     * @param response_neighbour_terms - An array of objects containing neighbour term data retrieved from the response.
     * Each object has properties: term, distance, and ponderation.
     */
    constructor(queryTermValue: string, id: string, title: string, abstract: string, response_neighbour_terms: any[]){
        this.queryTerm = new QueryTerm(queryTermValue)
        this.id = id
        this.title = title
        this.abstract = abstract
        this.initializeNeighbourTermsFromResponse(response_neighbour_terms)
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

    public toObject(): DocumentObject {
        return {
            doc_id: this.id,
            title: this.title,
            abstract: this.abstract,
            neighbour_terms: this.queryTerm.getNeighbourTermsAsObjects()
        }
    }

    /**
     * Initializes neighbour terms from the response data.
     *
     * @param response_neighbour_terms - An array of objects containing neighbour term data.
     * Each object has properties: term, distance, and ponderation.
     *
     * 
     *
     * @remarks
     * This function iterates over the response data, creates new NeighbourTerm instances for each term object,
     * and adds them to the QueryTerm's neighbour terms list.
     */
    private initializeNeighbourTermsFromResponse(response_neighbour_terms: any[]): void {
        const neighbourTerms = []
        for (const termObject of response_neighbour_terms) {
            neighbourTerms.push(new NeighbourTerm(this.queryTerm, termObject.term, termObject.distance, termObject.ponderation))
        }
        this.queryTerm.setNeighbourTerms(neighbourTerms)
    }
    
}

interface RankingObject {
    visible_neighbour_terms: NTermObject[];
    documents: DocumentObject[];
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

    /**
     * Reorders the documents in the ranking based on the provided positions array.
     * If the lengths of the positions array and the documents array do not match,
     * logs an error message to the console and returns without modifying the documents.
     *
     * @param positions - An array of integers representing the new order of the documents.
     * Each integer corresponds to the index of a document in the documents array.
     *
     * 
     *
     * @remarks
     * This function iterates over the positions array and creates a new array of documents
     * in the new order specified by the positions array.
     * It then assigns the reorderedDocuments array back to the documents property of the Ranking instance.
     */
    public reorderDocuments(positions: number[]): void {
        if (positions.length !== this.documents.length) {
            console.log('Positions array length must match documents array length.');
            return
        }
        const reorderedDocuments = new Array(this.documents.length);
        for (let i = 0; i < positions.length; i++) {
            reorderedDocuments[i] = this.documents[positions[i]];
        }
        this.documents = reorderedDocuments;
    }

    public toObject(): RankingObject {
        return {
            visible_neighbour_terms: this.visibleQueryTerm.getNeighbourTermsAsObjects(),
            documents: this.documents.map(document => document.toObject())
        }
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

    /**
     * Removes a neighbour term from the visible query term.
     * 
     * This function retrieves the neighbour term associated with the provided id,
     * and if found, removes it from the visible query term's neighbour terms list.
     * It then updates the neighbour terms table in the query service.
     * 
     * @param id - The id of the neighbour term to be removed.
     * 
     */
    public removeNeighbourTerm(id: string): void {
        const neighbourTerm = this.getVisibleQueryTerm().getNeighbourTermById(id)
        if (neighbourTerm === undefined) return
        this.getVisibleQueryTerm().removeNeighbourTerm(neighbourTerm)
        this.queryService.updateNeighbourTermsTable()
    }

    /**
     * Centers the graph on the CentralNode.
     * 
     * This function is responsible for zooming in the graph and centering it on the CentralNode.
     * It first zooms in the graph by a factor of 1.2, then checks if the visible query term has a node.
     * If the node exists and is a CentralNode, it centers the graph on the node.
     * 
     * 
     */
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

    /**
     * This function generates visible neighbour terms for the current query term.
     * 
     * @param result - The result object containing neighbour terms data.
     * The result object is expected to have a property 'visible_neighbour_terms',
     * which is an array of objects representing neighbour terms.
     * Each object should have properties 'term', 'distance', and 'ponderation'.
     * 
     * It iterates over the neighbour terms in the result, creates new NeighbourTerm instances,
     * and adds them to the QueryTerm's neighbour terms list.
     */
    private generateVisibleNeighbourTerms(result: any) {
        // Iterate over the neighbour terms in the result
        for (let termObject of result['visible_neighbour_terms']) {
            // Create a new NeighbourTerm instance for each term object
            const neighbourTerm = new NeighbourTerm(this.getVisibleQueryTerm(), termObject.term, termObject.distance, termObject.ponderation)

            // Add the neighbour term to the QueryTerm's neighbour terms list
            this.addNeighbourTerm(neighbourTerm)
        }
    }

    /**
     * Generates ranking documents for the current query term.
     * 
     * @param result - The result object containing ranking documents data.
     * The result object is expected to have a property 'documents',
     * which is an array of objects representing documents.
     * Each object should have properties 'doc_id', 'title', 'abstract', and 'neighbour_terms'.
     * 
     * It iterates over the documents in the result, creates new Document instances,
     * and adds them to the QueryTerm's ranking documents list.
     */
    private generateRankingDocuments(result: any) {
        // Iterate over the documents in the result
        for (let documentObject of result['documents']) {
            const doc_id = documentObject['doc_id']
            const title = documentObject['title']
            const abstract = documentObject['abstract']
            const response_neighbour_terms = documentObject['neighbour_terms']
            const document = new Document(this.ranking.getVisibleQueryTerm().getValue(), doc_id, title, abstract, response_neighbour_terms)
            this.addDocument(document)
        }
    }

    /**
     * Adds a neighbour term to the QueryTerm's neighbour terms list.
     * It also updates the neighbour terms table in the QueryService.
     * If the QueryTerm is currently visible, it displays the views of the neighbour term.
     *
     * @param neighbourTerm - The neighbour term to be added.
     */
    private addNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        this.getVisibleQueryTerm().addNeighbourTerm(neighbourTerm)
        this.queryService.updateNeighbourTermsTable()
        if (this.isVisible) this.display()
    }
    
    /**
     * Adds a document to the ranking and updates the results list.
     * 
     * @param document - The document to be added to the ranking.
     * 
     * @remarks
     * This function is responsible for adding a new document to the ranking and updating the results list.
     * It calls the `addDocument` method of the ranking and the `updateResultsList` method of the query service.
     */
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
     * Updates the table with neighbour terms data.
     * 
     * This function retrieves the table body element, clears existing rows, and then iterates over the neighbour terms of the active query term.
     * For each neighbour term, it creates a new row in the table and populates the cells with the term's value and hops.
     * 
     * @remarks
     * This function assumes that the table body element is already present in the HTML structure.
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

    /**
     * Updates the list of query results with new documents.
     * Clears existing list items in the results list before adding new ones.
     * 
     * @remarks
     * This function is responsible for populating the results list with the documents retrieved from the active query term.
     * It iterates over the documents in the ranking and creates list items for each one, including title and abstract elements.
     * Click event listeners and mouse event listeners are added to the title elements to handle user interactions.
     * 
     */
    public updateList(): void {
        // Clear existing list items
        this.dynamicList.innerHTML = '';

        // Check if the activeTermService is defined
        if (this.activeTermService === undefined) return

        // Get the ranking of the active query term
        let documents = this.activeTermService.getRanking().getDocuments()
    
        for (let i = 0; i < documents.length; i++) {
            // Create a new list item, title and abstract elements
            const listItem = document.createElement('li');
            const titleElement = this.createTitleElement(i, documents[i])
            const abstractElement = this.createAbstractElement(documents[i])
    
            // Add a click event listener and mouse event listeners to the title element
            this.addEventListenersToTitleElement(titleElement, abstractElement)
    
            // Append the title and abstract to the list item
            listItem.appendChild(titleElement);
            listItem.appendChild(abstractElement);
    
            // Append the list item to the dynamic list container
            this.dynamicList.appendChild(listItem);
        }
    }

    /**
     * Creates a title element for a document.
     * 
     * @param index - The index of the document in the list.
     * @param doc - The document for which to create the title element.
     * 
     * @returns A new HTMLSpanElement representing the title of the document.
     */
    private createTitleElement(index: number, doc: Document): HTMLSpanElement {
        const titleElement = document.createElement('span');
        titleElement.className = 'title';
        titleElement.textContent = (index + 1) + ". " + doc.getTitle();
        // Highlight the title element with orange color for the query terms and yellow color for the neighbour terms
        this.applyHighlighting(titleElement)
        return titleElement;
    }

    /**
     * Creates an abstract element for a document.
     * 
     * @param doc - The document for which to create the abstract element.
     * 
     * @returns A new HTMLParagraphElement representing the abstract of the document.
     */
    private createAbstractElement(doc: Document): HTMLParagraphElement {
        const abstractElement = document.createElement('p');
        abstractElement.className = 'abstract';
        abstractElement.textContent = doc.getAbstract();
        // Highlight the abstract element with orange color for the query terms and yellow color for the neighbour terms
        this.applyHighlighting(abstractElement)
        abstractElement.style.display = "none";
        return abstractElement;
    }

    /**
     * Applies highlighting to the words in an HTML element.
     * 
     * @param element - The HTML element to apply highlighting to.
     */
    private applyHighlighting(element: HTMLElement): void {
        const queryTerms = this.activeTermService?.getVisibleQueryTerm().getValue() as string
        const queryTermsList = TextUtils.separateBooleanQuery(queryTerms)
        const neighbourTermsList = this.activeTermService?.getVisibleQueryTerm().getNeighbourTermsValues() as string[]
        this.applyHighlightingToWords(element, queryTermsList, 'orange');
        this.applyHighlightingToWords(element, neighbourTermsList, 'yellow');
    }

    /**
     * Applies highlighting to specific words in an HTML element.
     * 
     * @param element - The HTML element to apply highlighting to.
     * @param words - The words to highlight.
     * @param color - The color to use for highlighting.
     */
    private applyHighlightingToWords(element: HTMLElement, words: string[], color: string = 'orange'): void {
        const originalText = element.innerHTML;
        const highlightedText = this.highlightWords(originalText, words, color);
        element.innerHTML = highlightedText;
    }

    /**
     * Highlights specific words in a given text string.
     * 
     * @param text - The original text string.
     * @param words - An array of words to highlight.
     * @param color - The color to use for highlighting. Default is 'orange'.
     * 
     * @returns A new string with the specified words highlighted with the given color.
     * Words are highlighted by wrapping them in a span element with the specified background color.
     */
    private highlightWords(text: string, words: string[], color: string = 'orange'): string {
        // Create a regular expression to find words that contain any substring of 'words'
        const regex = new RegExp(words.join('|'), 'gi');
    
        // Split text by spaces and replace matching words
        const highlightedText = text.split(' ').map(word => {
            if (regex.test(word)) {
                return `<span style="background-color: ${color};">${word}</span>`;
            }
            return word;
        }).join(' ');
    
        return highlightedText;
    }

    /**
     * This function adds a click event listener to the title element, which opens the original URL document webpage in a new tab when clicked.
     * It also adds mouseenter and mouseleave event listeners to change the title's color and cursor style.
    */
    private addEventListenersToTitleElement(titleElement: HTMLSpanElement, abstractElement: HTMLParagraphElement): void {
        titleElement.addEventListener('click', () => {
            // When the list item is clicked, opens the original URL document webpage in a new tab
            //window.open('https://ieeexplore.ieee.org/document/' + documents[i].getId(), '_blank');
            if (abstractElement.style.display === "none") {
                abstractElement.style.display = "";
            } else {
                abstractElement.style.display = "none";
            }
        });

        titleElement.addEventListener("mouseenter", () => {
            titleElement.style.color = "darkblue";
            titleElement.style.cursor = "pointer";
        });
    
        titleElement.addEventListener("mouseleave", () => {
            titleElement.style.color = "black";
        });
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

    /**
     * Constructs a new instance of RerankComponent.
     * 
     * @param queryService - The QueryService instance to be used for reranking operations.
     */
    constructor(queryService: QueryService) {
        this.queryService = queryService
        this.button = document.getElementById('rerankButton') as HTMLButtonElement

        // Add event listener to the button element
        this.button.addEventListener('click', this.handleRerankClick.bind(this))
    }

    /**
     * Handles the reranking button click event.
     * 
     * If the active QueryTermService exists, it creates a ranking object,
     * sends a POST request to the reranking endpoint, and handles the response.
     * 
     * @remarks
     * This method is asynchronous and uses the await keyword to handle the POST request.
     */
    private async handleRerankClick() {
        if (this.queryService.getActiveQueryTermService() !== undefined) {
            // Create the data to be sent in the POST request
            const ranking = this.queryService.getActiveQueryTermService()?.getRanking().toObject()

            // Send the POST request
            const response = await HTTPRequestUtils.postData('rerank', ranking)
            
            if (response) {
                // Handle the response accordingly
                const ranking_new_positions: number[] = response['ranking_new_positions'] 
                this.queryService.getActiveQueryTermService()?.getRanking().reorderDocuments(ranking_new_positions)
                this.queryService.updateResultsList()
            }
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
const queryComponent: QueryComponent = new QueryComponent(queryService)
const rerankComponent: RerankComponent = new RerankComponent(queryService)


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
