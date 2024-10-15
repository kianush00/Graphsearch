import cytoscape from "cytoscape";
import $ from 'jquery';

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

        return parseFloat(this.calculateEuclideanDistance(dx, dy).toFixed(0))
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
        const randomDistance = this.getRandomFloat(0.0, 200.0)
        const randomAngle = this.getRandomAngle()
        return this.getAngularPosition(randomAngle, randomDistance)
    }

    /**
     * A static method that generates a random angle in radians.
     *
     * @returns {number} A random angle between 0 (inclusive) and 2pi (exclusive).
     */
    public static getRandomAngle(): number {
        return this.getRandomFloat(0.0, 2 * Math.PI)
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
        const randomAngle = this.getRandomAngle()
        return this.getAngularPosition(randomAngle, distance)
    }

    /**
     * Generates a random floating-point number within a specified range.
     *
     * @param min - The minimum value (inclusive) of the range.
     * @param max - The maximum value (exclusive) of the range.
     *
     * @returns A random floating-point number within the range [min, max).
     *
     * @remarks
     * This function uses the `window.crypto.getRandomValues` method to generate a random number.
     * It then scales the random number to fit within the specified range and returns the result.
     */
    public static getRandomFloat(min: number, max: number): number {
        const randomBuffer = new Uint32Array(1);
        window.crypto.getRandomValues(randomBuffer);
        const randomNumber = randomBuffer[0] / (0xFFFFFFFF + 1); // 0xFFFFFFFF is the max value for Uint32
        return randomNumber * (max - min) + min;
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
            result += charsList.charAt(Math.floor(MathUtils.getRandomFloat(0.0, charsList.length)))
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


class ConversionUtils {
    private static readonly minMaxDistancesUserGraph: [number, number] = [45.0, 125.0]
    private static readonly minMaxDistancesSentenceGraph: [number, number] = [40.0, 65.0]
    private static readonly hopMinValue: number = 1.0

    /**
     * Converts the number of hops to the corresponding distance in the graph.
     *
     * @param hops - The number of hops from the central node to the neighbour term.
     * @param hopMaxValue - The maximum number of hops allowed in the graph.
     *
     * @returns {number} - The distance from the central node to the neighbour term.
     * The distance is calculated based on the normalized value of the hops,
     * using a linear transformation between the minimum and maximum distances.
     *
     * @remarks
     * This function assumes that the minimum and maximum distances are defined.
     * It also assumes that the hopMinValue is 0.
     * If the hopMaxValue is less than 2, the function returns 1.0.
     */
    public static convertHopsToDistance(hops: number, hopMaxValue: number, userGraphConversion: boolean): number {
        if (hopMaxValue < 2) return 1.0
        const minMaxDistances = this.getMinMaxDistances(userGraphConversion)
        const normalizedValue = this.normalize(hops, this.hopMinValue, hopMaxValue, minMaxDistances[0], minMaxDistances[1])
        return normalizedValue
    }

    /**
     * Converts a given distance to hops, based on a normalized value within a specified range.
     *
     * @param distance - The distance to be converted to hops.
     * @param hopMaxValue - The maximum number of hops that can be achieved.
     *
     * @returns {number} - The number of hops corresponding to the given distance, normalized within the specified range.
     *
     * @remarks
     * This function normalizes the given distance within the range of minimum and maximum distances,
     * and then maps the normalized value to the range of minimum and maximum hops.
     * If the hopMaxValue is less than 2, the function returns 1.0.
     * The returned value is rounded to one decimal place.
     */
    public static convertDistanceToHops(distance: number, hopMaxValue: number, userGraphConversion: boolean): number {
        if (hopMaxValue < 2) return 1.0
        const minMaxDistances = this.getMinMaxDistances(userGraphConversion)
        const normalizedValue = this.normalize(distance, minMaxDistances[0], minMaxDistances[1], this.hopMinValue, hopMaxValue)
        return parseFloat(normalizedValue.toFixed(1))
    }

    /**
     * Validates if the provided distance is out of the specified range.
     *
     * @param distance - The distance to be validated.
     *
     * @returns {boolean} - Returns `true` if the distance is out of range, `false` otherwise.
     *
     * @remarks
     * This function checks if the provided distance is less than the minimum distance or greater than the maximum distance.
     * If either condition is met, the function returns `true`, indicating that the distance is out of range.
     * Otherwise, it returns `false`, indicating that the distance is within the specified range.
     */
    public static validateDistanceOutOfRange(distance: number) : boolean {
        return distance < this.minMaxDistancesUserGraph[0] || distance > this.minMaxDistancesUserGraph[1]
    }

    /**
     * Normalizes a given value within a specified range.
     *
     * @param value - The value to be normalized.
     * @param oldMin - The minimum value of the original range.
     * @param oldMax - The maximum value of the original range.
     * @param newMin - The minimum value of the new range.
     * @param newMax - The maximum value of the new range.
     *
     * @returns The normalized value within the new range.
     * If the input value is less than oldMin or greater than oldMax, the function returns 1.0.
     *
     * @remarks
     * This function normalizes a given value by scaling it from the original range to the new range.
     * It uses a linear transformation formula to calculate the normalized value.
     */
    private static normalize(value: number, oldMin: number, oldMax: number, newMin: number, newMax: number): number {
        if (value < oldMin) {
            return newMin
        } else if (value > oldMax) {
            return newMax
        }

        const normalizedValue = newMin + ((value - oldMin) * (newMax - newMin)) / (oldMax - oldMin);
        return normalizedValue;
    }

    private static getMinMaxDistances(userGraphConversion: boolean) {
        return userGraphConversion ? this.minMaxDistancesUserGraph : this.minMaxDistancesSentenceGraph
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

            // Check the status code of the response
            const statusCode = response.status;
            console.log('HTTP Status Code:', statusCode);
        
            // Handle the response if code is successful
            const result = await response.json();
            if (response.ok) { // response.ok is true if code is between 200-299
                console.log('Success:', result);
                const sizeKb = new TextEncoder().encode(JSON.stringify(result)).length / 1024;
                console.log('Size of response in KB:', sizeKb);
                return result;
            } else {
                console.error('Error:', result);
                alert(`${result['detail']}`);
                return null;
            }
        } catch (error) {
            console.error('Error:', error)
        }
    }

}



interface EdgeData {
    id: string
    source: string
    target: string
    distance?: number
}

class Edge {
    private readonly id: string
    private readonly sourceNode: GraphNode
    private readonly targetNode: GraphNode
    private distance: number
    private readonly hopLimit: number
    private readonly isUserGraphEdge: boolean
    private readonly cyElement: cytoscape.Core

    /**
     * Represents an edge in the graph, connecting two nodes.
     * 
     * @param sourceNode - The source node of the edge.
     * @param targetNode - The target node of the edge.
     * @param hopLimit - The maximum number of hops allowed for the edge.
     */
    constructor(sourceNode: GraphNode, targetNode: GraphNode, hopLimit: number, isUserGraphEdge: boolean) {
        this.id = "e_" + targetNode.getId()
        this.sourceNode = sourceNode
        this.targetNode = targetNode
        this.distance = MathUtils.getDistanceBetweenNodes(sourceNode, targetNode)
        this.hopLimit = hopLimit
        this.isUserGraphEdge = isUserGraphEdge
        this.cyElement = isUserGraphEdge ? cyUser : cySentence;
        this.addVisualEdgeToInterface()
    }

    /**
     * Sets the distance for the edge between the source and target nodes.
     * This method updates the distance value and the corresponding edge data in the graph.
     *
     * @param distance - The new distance for the edge.
     *
     * @remarks
     * This method assumes that the source and target nodes are already associated with the edge.
     * It also assumes that the graph is represented using the Cytoscape.js library.
     *
     * @returns {void} - This function does not return any value.
     */
    public setDistance(distance: number): void {
        this.distance = distance;
        let cyEdge = this.cyElement.edges(`[source = "${this.sourceNode.getId()}"][target = "${this.targetNode.getId()}"]`)
        const hops = ConversionUtils.convertDistanceToHops(this.distance, this.hopLimit, this.isUserGraphEdge)
        if (!this.isUserGraphEdge) {
            cyEdge.data('distance', hops)
        }
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
        this.cyElement.remove(this.cyElement.getElementById(this.id))
    }

    /**
     * Converts the Edge instance into a serializable object.
     * 
     * @returns An object containing the data of the Edge.
     * The object has properties: id, source, target, and distance.
     * The distance is converted to hops using the ConversionUtils.
     */
    public toObject(): { data: EdgeData } {
        const baseData = {
            id: this.id,
            source: this.sourceNode.getId(),
            target: this.targetNode.getId(),
        };
    
        // Add distance to the data if the edge is not a user graph edge
        if (!this.isUserGraphEdge) {
            return {
                data: {
                    ...baseData,
                    distance: ConversionUtils.convertDistanceToHops(
                        this.distance, 
                        this.hopLimit, 
                        this.isUserGraphEdge
                    ),
                }
            };
        }
    
        // Return the base data if the edge is a user graph edge
        return { data: baseData };
    }

    private addVisualEdgeToInterface(): void {
        this.cyElement.add(this.toObject())
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
    protected cyElement: cytoscape.Core
    
    /**
     * Represents a node in the graph.
     * 
     * @param id - The unique identifier of the node.
     * @param label - The label of the node.
     * @param position - The position of the node in the graph.
     * @param type - The type of the node.
     */
    constructor(id: string, label: string, position: Position, type: NodeType, isUserGraphNode: boolean) {
        this.id = id
        this.label = label
        this.position = position
        this.type = type
        this.cyElement = isUserGraphNode ? cyUser : cySentence;
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
        this.cyElement.getElementById(this.id).data('label', label)
    }

    /**
     * Removes the visual node from the graph interface.
     *
     * @remarks
     * This function removes the node with the given ID from the graph interface.
     * It uses the Cytoscape.js library to select the node by its ID and remove it from the graph.
     */
    public remove(): void {
        this.cyElement.remove(this.cyElement.getElementById(this.id))
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
    constructor(id: string, x: number, y: number, isUserGraphNode: boolean) {
        let _id = id
        let _label = id
        let _position = { x, y }
        let _type = NodeType.central_node
        super(_id, _label, _position, _type, isUserGraphNode)
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
        this.cyElement.add(this.toObject()).addClass(this.type.toString()).lock().ungrabify()
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
    constructor(id: string, isUserGraphNode: boolean, distance: number = 0) {
        let _id = id
        let _label = id
        // Generates a random angle from the provided distance, to calculate the new position
        let _position = MathUtils.getAngularPosition(MathUtils.getRandomAngle(), distance)
        let _type = NodeType.outer_node
        super(_id, _label, _position, _type, isUserGraphNode)
        this.addVisualNodeToInterface(isUserGraphNode)
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
        this.cyElement.getElementById(this.id).position(this.position)
    }

    /**
     * Adds a visual node to the graph interface.
     * 
     * This function creates a new node in the graph using the `toObject` method,
     * which returns an object containing the node's data and position.
     * The node is then added to the graph using the `cy.add` method.
     * The node's class is set to the string representation of the node's type.
     */
    private addVisualNodeToInterface(isUserGraphNode: boolean): void {
        const element = this.cyElement.add(this.toObject()).addClass(this.type.toString());
        if (!isUserGraphNode) element.ungrabify();
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
    proximity_ponderation: number;
    total_ponderation: number;
    criteria: string;
    distance?: number;
}


class NeighbourTerm extends Term implements ViewManager {
    protected node: OuterNode | undefined
    private readonly queryTerm: QueryTerm
    private hops: number
    private nodePosition: Position = { x: 0, y: 0 }
    private edge: Edge | undefined
    private readonly proximityPonderation: number
    private readonly totalPonderation: number
    private criteria: string
    private readonly hopLimit: number

    /**
     * Represents a neighbour term in the graph.
     * It manages the associated views, such as the OuterNode and Edge.
     * 
     * @param queryTerm - The QueryTerm associated with this neighbour term.
     * @param value - The value of the neighbour term.
     * @param hops - The number of hops from the central node to this neighbour term.
     * @param ponderation - The ponderation of this neighbour term.
     * @param hopLimit - The maximum number of hops allowed for the neighbour term.
     */
    constructor(queryTerm: QueryTerm, value: string, hops: number, proximityPonderation: number, 
        totalPonderation: number, criteria: string, hopLimit: number) {
        super(value)
        this.queryTerm = queryTerm
        this.proximityPonderation = proximityPonderation
        this.totalPonderation = totalPonderation
        this.criteria = criteria
        this.hops = this.queryTerm.getIsUserQuery() ? 1.0 : hops
        this.hopLimit = hopLimit
        this.setLabel(value)
    }

    /**
     * Displays the views of the neighbour term in the graph.
     * This includes creating and positioning the OuterNode and Edge.
     */
    public displayViews(): void {
        const isUserQuery = this.queryTerm.getIsUserQuery()
        // If the term is from a visible sentence, then the sentence graph doesn't display non proximity neighbour terms
        if ((!isUserQuery) && (this.criteria !== "proximity")) return

        // Build the outer node and its edge, and display them
        this.node = new OuterNode(TextUtils.getRandomString(24), isUserQuery)
        this.node.setPosition(this.nodePosition)
        this.node.setLabel(this.value)
        if (this.queryTerm.getNode() === undefined) return 
        this.edge = new Edge(this.queryTerm.getNode() as CentralNode, this.node, this.hopLimit, isUserQuery)
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

    public setHops(hops: number): void {
        this.hops = hops
        if (this.queryTerm.getIsUserQuery()) {
            this.updateUserCriteria(hops)
        }
    }

    public getProximityPonderation(): number {
        return this.proximityPonderation
    }

    public getTotalPonderation(): number {
        return this.totalPonderation
    }

    public getCriteria(): string {
        return this.criteria
    }

    public getHopLimit(): number {
        return this.hopLimit
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
        const baseData = {
            term: this.value,
            proximity_ponderation: this.proximityPonderation,
            total_ponderation: this.totalPonderation,
            criteria: this.criteria
        };

        if (this.queryTerm.getIsUserQuery()) {  // If it's a user neighbour term
            return baseData
        } else {    // If it's not a user neighbour term
            return {
                ...baseData,
                distance: this.hops
            }
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
        const hops = ConversionUtils.convertDistanceToHops(distance, this.hopLimit, this.queryTerm.getIsUserQuery())
        this.setHops(hops)
        this.updateNodePosition(distance)
    }

    /**
     * Updates the position of the neighbour term node and updates the neighbour term's hops.
     *
     * @param neighbourTermsLength - The total number of neighbour terms associated with the query term.
     * @param index - The index of the neighbour term in the neighbour terms list.
     *
     * @returns {void} - This function does not return any value.
     * 
     * @remarks
     * This function calculates the new angle for the neighbour term node based on the index and the total number of neighbour terms.
     * It then calculates the distance from the central node using the neighbour term's hops.
     * The new position is obtained using the calculated angle and distance.
     * Finally, it updates the position of the neighbour term node and calls the `updateNodePosition` method.
     */
    public updateSymmetricalAngularPosition(neighbourTermsLength: number, index: number): void {
        const newAngle = (index / neighbourTermsLength) * Math.PI * 2 + 0.25
        const nodeDistance = ConversionUtils.convertHopsToDistance(this.hops, this.hopLimit, this.queryTerm.getIsUserQuery())
        this.nodePosition = MathUtils.getAngularPosition(newAngle, nodeDistance)
        this.updateNodePosition(nodeDistance)
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
            if (ConversionUtils.validateDistanceOutOfRange(positionDistance)) {
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
     * Updates the position of the neighbour term node and the neighbour term's hops.
     *
     * @param distance - The distance from the central node to the neighbour term node.
     * This distance is used to update the neighbour term's hops and the position of the neighbour term node.
     *
     * @returns {void} - This function does not return any value.
     */
    private updateNodePosition(distance: number): void {
        this.node?.setPosition(this.nodePosition)
        this.edge?.setDistance(distance)
    }

    /**
    * Updates the criteria of the neighbour term based on the number of hops.
    *
    * @param hops - The number of hops from the central node to the neighbour term.
    *
    * @returns {void} - This function does not return any value.
    *
    * @remarks
    * This function checks the number of hops and updates the criteria of the neighbour term accordingly.
    * If the number of hops is less than 1.7, the criteria is set to "proximity".
    * If the number of hops is between 1.7 and 3.2 (exclusive), the criteria is set to "frequency".
    * If the number of hops is greater than or equal to 3.2, the criteria is set to "exclusion".
    */
    private updateUserCriteria(hops: number): void {
        if (hops < 1.7) {
            this.criteria = "proximity";
        } else if (hops < 3.2) {
            this.criteria = "frequency";
        } else {
            this.criteria = "exclusion";
        }
    }

}


/**
 * Represents a query term that is associated with a central node in the graph.
 * It also manages neighbour terms related to the query term.
 */
class QueryTerm extends Term implements ViewManager {
    protected node: CentralNode | undefined
    private neighbourTerms: NeighbourTerm[] = []
    private readonly isUserQuery: boolean

    constructor(value: string, isUserQuery: boolean) {
        super(value);
        this.isUserQuery = isUserQuery;
    }

    public getIsUserQuery(): boolean {
        return this.isUserQuery;
    }

    /**
     * Displays the views of the query term and its associated neighbour terms in the graph.
     * This includes creating and positioning the CentralNode and OuterNodes.
     */
    public displayViews(): void {
        this.node = new CentralNode(this.value, 0, 0, this.isUserQuery)
        for (let neighbourTerm of this.neighbourTerms) {
            neighbourTerm.displayViews();
        }
        this.centerNode();
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
        this.updateOuterNodesAngles()
    }

    public getNeighbourTermsValues(): string[] {
        return this.neighbourTerms.map(term => term.getValue())
    }

    public getNeighbourTermsAsObjects(): NTermObject[] {
        return this.neighbourTerms.map(term => term.toObject())
    }

    public getNeighbourTermByNodeId(id: string): NeighbourTerm | undefined {
        return this.neighbourTerms.find(nterm => nterm.getNode()?.getId() === id)
    }

    public getNeighbourTermByValue(value: string): NeighbourTerm | undefined {
        return this.neighbourTerms.find(nterm => nterm.getValue() === value)
    }

    public addNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        this.neighbourTerms.push(neighbourTerm)
        this.updateOuterNodesAngles()
    }

    public removeNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        this.neighbourTerms = this.neighbourTerms.filter(term => term !== neighbourTerm)
        neighbourTerm.removeViews()
        this.updateOuterNodesAngles()
    }

    private updateOuterNodesAngles(): void {
        for (let i = 0; i < this.neighbourTerms.length; i++) {
            this.neighbourTerms[i].updateSymmetricalAngularPosition(this.neighbourTerms.length, i)
        }
    }

    /**
     * Centers the graph on the CentralNode.
     * 
     * This function is responsible for zooming in the graph and centering it on the CentralNode.
     * It first zooms in the graph by a factor of 1.2, then checks if the visible query term has a node.
     * If the node exists and is a CentralNode, it centers the graph on the node.
     */
    private centerNode(): void {
        const cyElement = this.isUserQuery ? cyUser : cySentence
        cyElement.zoom(1.2)
        if (this.node === undefined) return
        cyElement.center(cyElement.getElementById(this.node.getId()))
    }
}


class TextElement {
    protected queryTerm: QueryTerm

    /**
    ​ * Constructor for the QueryTerm class.
    ​ * Initializes a new QueryTerm instance with the provided query term value and neighbour terms.
    ​ *
    ​ * @param queryTermValue - The value of the query term.
    ​ * @param responseNeighbourTerms - An array of objects representing neighbour terms retrieved from the response.
    ​ * Each object should have properties: term, distance, and ponderation.
    ​ * @param hopLimit - The maximum number of hops allowed for the neighbour terms in the document.
    ​ */
    constructor(queryTermValue: string, responseNeighbourTerms: any[], hopLimit: number) {
        this.queryTerm = new QueryTerm(queryTermValue, false)
        this.initializeNeighbourTermsFromResponse(responseNeighbourTerms, hopLimit)
    }

    public getQueryTerm(): QueryTerm {
        return this.queryTerm
    }

    /**
     * Initializes neighbour terms from the response data.
     * 
     * @param responseNeighbourTerms - An array of objects containing neighbour term data retrieved from the response.
     * Each object has properties: term, distance, and ponderation.
     * 
     * @param hopLimit - The maximum number of hops allowed for the neighbour terms in the document.
     * 
     * @returns {void} - This function does not return any value.
     */
    private initializeNeighbourTermsFromResponse(responseNeighbourTerms: any[], hopLimit: number): void {
        const neighbourTerms = []
        for (const termObject of responseNeighbourTerms) {
            let neighbourTerm = new NeighbourTerm(this.queryTerm, termObject.term, termObject.distance, 
                termObject.proximity_ponderation, termObject.total_ponderation, termObject.criteria, hopLimit)
            neighbourTerms.push(neighbourTerm)
        }
        this.queryTerm.setNeighbourTerms(neighbourTerms)
    }
}



interface SentenceObject {
    position_in_doc: number;
    raw_text: string;
    all_neighbour_terms: NTermObject[];
}

class Sentence extends TextElement {
    private readonly positionInDoc: number
    private readonly rawText: string

    /**
    ​ * Constructor for the Sentence class.
    ​ * Initializes a new Sentence instance with the provided query term value, neighbour terms, and sentence details.
    ​ *
    ​ * @param queryTermValue - The value of the query term associated with the sentence.
    ​ * @param responseNeighbourTerms - An array of objects representing neighbour terms retrieved from the response.
    ​ * Each object should have properties: term, distance, and ponderation.
    ​ * @param hopLimit - The maximum number of hops allowed for the neighbour terms in the document.
    ​ * @param positionInDoc - The position of the sentence in the document.
    ​ * @param rawText - The raw text of the sentence.
    ​ */
    constructor(queryTermValue: string, responseNeighbourTerms: any[], hopLimit: number, positionInDoc: number, rawText: string){
        super(queryTermValue, responseNeighbourTerms, hopLimit)
        this.positionInDoc = positionInDoc
        this.rawText = rawText
    }

    public getPositionInDoc(): number {
        return this.positionInDoc
    }

    public getRawText(): string {
        return this.rawText
    }

    public toObject(): SentenceObject {
        return {
            position_in_doc: this.positionInDoc,
            raw_text: this.rawText,
            all_neighbour_terms: this.queryTerm.getNeighbourTermsAsObjects()
        }
    }
}


interface DocumentObject {
    doc_id: string;
    title: string;
    abstract: string;
    preprocessed_text: string;
    weight: number;
    all_neighbour_terms: NTermObject[];
}

class Document extends TextElement {
    private readonly id: string
    private readonly title: string
    private readonly abstract: string
    private readonly preprocessed_text: string
    private readonly weight: number
    private readonly sentences: Sentence[] = []

    /**
    ​ * Constructor for the Document class.
    ​ * Initializes a new Document instance with the provided query term value, neighbour terms, document details, and sentence data.
    ​ *
    ​ * @param queryTermValue - The value of the query term associated with the document.
    ​ * @param responseNeighbourTerms - An array of objects representing neighbour terms retrieved from the response.
    ​ * Each object has properties: term, distance, and ponderation.
    ​ * @param hopLimit - The maximum number of hops allowed for the neighbour terms in the document.
    ​ * @param idTitleAbstractPreprcsdtext - An array containing the document's id, title, abstract and preprocessed text.
    ​ * @param weight - The weight of the document.
    ​ * @param responseSentences - An array of objects representing sentences retrieved from the response.
    ​ * Each object has properties: position_in_doc, raw_text, and neighbour_terms.
    ​ */
    constructor(queryTermValue: string, responseNeighbourTerms: any[], hopLimit: number, idTitleAbstractPreprcsdtext: [string, string, string, string], 
        weight: number, responseSentences: any[]){
        super(queryTermValue, responseNeighbourTerms, hopLimit)
        this.id = idTitleAbstractPreprcsdtext[0]
        this.title = idTitleAbstractPreprcsdtext[1]
        this.abstract = idTitleAbstractPreprcsdtext[2]
        this.preprocessed_text = idTitleAbstractPreprcsdtext[3]
        this.weight = weight
        this.sentences = this.initializeSentencesFromResponse(responseSentences, hopLimit)
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

    public getWeight(): number {
        return this.weight
    }

    public getSentences(): Sentence[] {
        return this.sentences
    }

    public toObject(): DocumentObject {
        return {
            doc_id: this.id,
            title: this.title,
            abstract: this.abstract,
            preprocessed_text: this.preprocessed_text,
            weight: this.weight,
            all_neighbour_terms: this.queryTerm.getNeighbourTermsAsObjects()
        }
    }

    /**
    ​ * Initializes sentences from the provided response data.
    ​ * 
    ​ * @param responseSentences - An array of objects representing sentences retrieved from the response.
    ​ * Each object has properties 'position_in_doc', 'raw_text', and 'neighbour_terms'.
    ​ * 
    ​ * @param hopLimit - The maximum number of hops allowed for the neighbour terms in the sentences.
    ​ * 
    ​ * @returns An array of Sentence instances, each representing a sentence from the response data.
    ​ * Each Sentence instance is created with the query term value, neighbour terms, hop limit, position, and raw text.
    ​ */
    private initializeSentencesFromResponse(responseSentences: any[], hopLimit: number): Sentence[] {
        const sentences = []
        for (const sentenceObject of responseSentences) {
            let sentence = new Sentence(this.queryTerm.getValue(), sentenceObject.all_neighbour_terms, 
                    hopLimit, sentenceObject.position_in_doc, sentenceObject.raw_text)
            sentences.push(sentence)
        }
        return sentences
    }
}


interface RankingObject {
    visible_neighbour_terms: NTermObject[];
    documents: DocumentObject[];
}

class Ranking {
    private readonly visibleQueryTerm: QueryTerm
    private readonly completeQueryTerm: QueryTerm
    private documents: Document[] = []
    private visibleSentence: Sentence | undefined

    constructor(queryTermValue: string) {
        this.visibleQueryTerm = new QueryTerm(queryTermValue, true)
        this.completeQueryTerm = new QueryTerm(queryTermValue, false)
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

    public getVisibleSentence(): Sentence | undefined {
        return this.visibleSentence
    }

    public setVisibleSentence(sentence: Sentence): void {
        this.visibleSentence?.getQueryTerm().removeViews()
        this.visibleSentence = sentence
        this.visibleSentence.getQueryTerm().displayViews()
    }

    /**
     * Reorders the documents in the ranking based on the provided positions array.
     * If the lengths of the positions array and the documents array do not match,
     * logs an error message to the console and returns without modifying the documents.
     *
     * @param positions - An array of integers representing the new order of the documents.
     * Each integer corresponds to the index of a document in the documents array.
     *
     * @remarks
     * This function iterates over the positions array and creates a new array of documents
     * in the new order specified by the positions array.
     * It then assigns the reorderedDocuments array back to the documents property of the Ranking instance.
     */
    public reorderDocuments(positions: number[]): void {
        if (positions.length !== this.documents.length) {
            console.log('Warning: Positions array length must match documents array length.');
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


/**
 * A service class responsible for managing query terms and their associated data.
 */
class QueryTermService {
    private readonly queryService: QueryService
    private readonly ranking: Ranking
    private isVisible: boolean = false

    constructor(queryService: QueryService, queryTermValue: string, searchResults: number, limitDistance: number, graphTerms: number) {
        this.queryService = queryService
        this.ranking = new Ranking(queryTermValue)
        this.retrieveData(searchResults, limitDistance, graphTerms)
    }

    public getVisibleQueryTerm(): QueryTerm {
        return this.ranking.getVisibleQueryTerm()
    }

    public getCompleteQueryTerm(): QueryTerm {
        return this.ranking.getCompleteQueryTerm()
    }

    public getRanking(): Ranking {
        return this.ranking
    }

    public getVisibleSentence(): Sentence | undefined {
        return this.ranking.getVisibleSentence()
    }

    public setVisibleSentence(sentence: Sentence): void {
        this.ranking.setVisibleSentence(sentence)
    }

    /**
     * If the node is dragged, updates the position of the neighbour term node and 
     * updates the neighbour term's hops.
     * @param id - The id of the neighbour term node.
     * @param position - The new position of the neighbour term node.
     */
    public nodeDragged(id: string, position: Position): void {
        let neighbourTerm = this.getVisibleQueryTerm().getNeighbourTermByNodeId(id)
        if (neighbourTerm === undefined) return
        neighbourTerm.setPosition(position)

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
    }

    /**
     * This method removes the visual nodes and edges from the graph interface.
     */
    public deactivate(): void {
        this.isVisible = false
        this.getVisibleQueryTerm().removeViews()
        this.getVisibleSentence()?.getQueryTerm().removeViews()
    }

    /**
     * Adds a neighbour term to the QueryTerm's neighbour terms list.
     * It also updates the neighbour terms table in the QueryService.
     * If the QueryTerm is currently visible, it displays the views of the neighbour term.
     *
     * @param neighbourTerm - The neighbour term to be added.
     */
    public addVisibleNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        if (this.ranking.getVisibleQueryTerm().getNeighbourTerms().length > 19) return
        this.getVisibleQueryTerm().addNeighbourTerm(neighbourTerm)
        this.queryService.updateNeighbourTermsTable()
        this.queryService.updateAddTermsTable()
        if (this.isVisible) this.display()
    }

    /**
     * Removes a neighbour term from the visible query term.
     * 
     * This function retrieves the neighbour term associated with the provided id,
     * and if found, removes it from the visible query term's neighbour terms list.
     * It then updates the neighbour terms table in the query service.
     * 
     * @param id - The id of the neighbour term to be removed.
     */
    public removeVisibleNeighbourTerm(id: string): void {
        let neighbourTerm = this.getVisibleQueryTerm().getNeighbourTermByNodeId(id)
        if (neighbourTerm === undefined || this.ranking.getVisibleQueryTerm().getNeighbourTerms().length < 2) return
        this.getVisibleQueryTerm().removeNeighbourTerm(neighbourTerm)
        this.queryService.updateNeighbourTermsTable()
        this.queryService.updateAddTermsTable()
        if (this.isVisible) this.display()
    }

    /**
     * Adds a neighbour term to the complete query term.
     * 
     * This function takes a NeighbourTerm instance as a parameter and adds it to the complete query term's neighbour terms list.
     * It also updates the add terms table in the QueryService.
     * 
     * @param neighbourTerm - The neighbour term to be added.
     * 
     * @returns {void} - This function does not return any value.
     */
    public addCompleteNeighbourTerm(neighbourTerm: NeighbourTerm): void {
        this.getCompleteQueryTerm().addNeighbourTerm(neighbourTerm)
    }

    /**
    * Changes the cursor type of the HTML document to the specified newCursorType.
    * This function is used to update the cursor style when hovering over a neighbour term node in the graph.
    * 
    * @param id - The unique identifier of the neighbour term node.
    * @param newCursorType - The new cursor type to be applied to the HTML document.
    * 
    * @returns {void} - This function does not return any value.
    */
    public changeCursorType(id: string, newCursorType: string): void {
        let neighbourTerm = this.getVisibleQueryTerm().getNeighbourTermByNodeId(id)
        if (neighbourTerm === undefined) return
        $('html,body').css('cursor', newCursorType);
    }

    /**
     * Retrieves data related to neighbour terms for the current query term.
     * The retrieved data is then used to create new NeighbourTerm instances,
     * which are added to the QueryTerm's neighbour terms list.
     * 
     * @param searchResults - The number of search results to retrieve.
     * @param limitDistance - The maximum distance limit for neighbour terms.
     * @param graphTerms - The number of neighbour terms to include in the graph.
     *
     * @returns {Promise<void>} - A promise that resolves when the data retrieval and processing are complete.
     */
    private async retrieveData(searchResults: number, limitDistance: number, graphTerms: number): Promise<void> {
        // Define the endpoint for retrieving neighbour terms data
        const endpoint = 'get-ranking'

        // Send a POST request to the endpoint with the query term value
        let _query = this.getVisibleQueryTerm().getValue()
        const data = {query: _query, search_results: searchResults, limit_distance: limitDistance, graph_terms: graphTerms}
        const result = await HTTPRequestUtils.postData(endpoint, data)

        // Check if the result is not null
        if (result) {
            this.generateVisibleNeighbourTerms(result, limitDistance)
            this.generateCompleteNeighbourTerms(result, limitDistance)
            this.generateRankingDocuments(result, limitDistance)
        }
    }

    /**
     * Generates visible neighbour terms for the current query term.
     * 
     * @param result - The result object containing neighbour terms data.
     * The result object is expected to have a property 'visible_neighbour_terms',
     * which is an array of objects representing neighbour terms.
     * Each object should have properties 'term', 'distance', and 'ponderation'.
     * 
     * @param hopLimit - The maximum number of hops allowed for the neighbour terms.
     * 
     * @returns {void} - This function does not return any value.
     */
    private generateVisibleNeighbourTerms(result: any, hopLimit: number): void {
        // Iterate over the neighbour terms in the result
        for (let termObject of result['visible_neighbour_terms']) {
            // Create a new NeighbourTerm instance for each term object
            let neighbourTerm = this.initializeNewNeighbourTerm(this.getVisibleQueryTerm(), termObject, hopLimit)

            // Add the neighbour term to the visible QueryTerm's neighbour terms list
            this.addVisibleNeighbourTerm(neighbourTerm)
        }
    }

    /**
     * Generates complete neighbour terms for the current query term.
     * 
     * @param result - The result object containing neighbour terms data.
     * The result object is expected to have a property 'complete_neighbour_terms',
     * which is an array of objects representing neighbour terms.
     * Each object should have properties 'term', 'distance', and 'ponderation'.
     * 
     * @param hopLimit - The maximum number of hops allowed for the neighbour terms.
     * 
     * @returns {void} - This function does not return any value.
     * It iterates over the neighbour terms in the result, creates a new NeighbourTerm instance for each term object,
     * and adds the neighbour term to the complete QueryTerm's neighbour terms list.
     */
    private generateCompleteNeighbourTerms(result: any, hopLimit: number): void {
        // Iterate over the neighbour terms in the result
        for (let termObject of result['complete_neighbour_terms']) {
            // Create a new NeighbourTerm instance for each term object
            let neighbourTerm = this.initializeNewNeighbourTerm(this.getCompleteQueryTerm(), termObject, hopLimit)

            // Add the neighbour term to the complete QueryTerm's neighbour terms list
            this.addCompleteNeighbourTerm(neighbourTerm)
        }

        // Update the neighbour terms table in the QueryService
        this.queryService.updateAddTermsTable()
    }

    /**
     * Initializes a new NeighbourTerm instance based on the provided term object and hop limit.
     * 
     * @param termObject - An object containing properties 'term', 'distance', and 'ponderation' representing a neighbour term.
     * @param hopLimit - The maximum number of hops allowed for the neighbour terms.
     * 
     * @returns A new NeighbourTerm instance with the provided term value, distance, ponderation, and hop limit.
     */
    private initializeNewNeighbourTerm(queryTerm: QueryTerm, termObject: any, hopLimit: number): NeighbourTerm {
        return new NeighbourTerm(queryTerm, termObject.term, termObject.distance, 
            termObject.proximity_ponderation, termObject.total_ponderation, termObject.criteria, hopLimit)
    }

    /**
     * Generates ranking documents for the current query term, and updates the results list component.
     * 
     * @param result - The result object containing ranking documents data.
     * The result object is expected to have a property 'documents',
     * which is an array of objects representing documents.
     * Each object should have properties 'doc_id', 'title', 'abstract', and 'neighbour_terms'.
     * 
     * @param hopLimit - The maximum number of hops allowed for the neighbour terms in the documents.
     * 
     * @returns {void} - This function does not return any value.
     */
    private generateRankingDocuments(result: any, hopLimit: number): void {
        // Iterate over the documents in the result
        for (let documentObject of result['documents']) {
            const doc_id = documentObject['doc_id']
            const title = documentObject['title']
            const abstract = documentObject['abstract']
            const preprocessed_text = documentObject['preprocessed_text']
            const weight = documentObject['weight']
            const response_neighbour_terms = documentObject['all_neighbour_terms']
            const sentences = documentObject['sentences']
            let document = new Document(this.ranking.getVisibleQueryTerm().getValue(), response_neighbour_terms, hopLimit, 
                    [doc_id, title, abstract, preprocessed_text], weight, sentences)
            this.addDocument(document)
        }

        // Update the ranking's documents list
        this.queryService.updateResultsList()
    }
    
    /**
     * Adds a document to the ranking and updates the results list.
     * 
     * @param document - The document to be added to the ranking.
     * 
     * @remarks
     * This function is responsible for adding a new document to the ranking and updating the results list.
     * It calls the `addDocument` method of the ranking
     */
    private addDocument(document: Document): void {
        this.getRanking().addDocument(document)
    }
}


class QueryService {
    private activeQueryTermService: QueryTermService | undefined
    private readonly queryTermServices: QueryTermService[]
    private readonly neighbourTermsTable: NeighbourTermsTable
    private readonly addTermsTable: AddTermsTable
    private readonly queryTermsList: QueryTermsList
    private readonly resultsList: ResultsList

    constructor() {
        this.queryTermServices = []
        this.neighbourTermsTable = new NeighbourTermsTable()
        this.resultsList = new ResultsList()
        this.queryTermsList = new QueryTermsList(this)
        this.addTermsTable = new AddTermsTable()
    }

    /**
     * Sets the query for the service.
     * Deactivates the currently active QueryTermService, creates a new Query object,
     * and triggers the query generation process.
     * 
     * @param queryValue - The new query string.
     * @param searchResults - The number of search results to retrieve.
     * @param limitDistance - The maximum distance limit for neighbour terms.
     * @param graphTerms - The number of neighbour terms to include in the graph.
     */
    public setQuery(queryValue: string, searchResults: number, limitDistance: number, graphTerms: number): void {
        this.activeQueryTermService?.deactivate()
        this.generateNewQueryTermService(queryValue, searchResults, limitDistance, graphTerms)
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
            this.addTermsTable.setActiveTermService(this.activeQueryTermService)
            this.resultsList.setActiveTermService(this.activeQueryTermService)
        }
    }

    public updateNeighbourTermsTable(): void {
        this.neighbourTermsTable.updateTable()
    }

    public updateResultsList(): void {
        this.resultsList.updateList()
    }

    public updateAddTermsTable(): void {
        this.addTermsTable.updateTable()
    }

    /**
     * Generates a new QueryTermService for a given query value.
     * This method checks if a QueryTermService for the given query value already exists.
     * If not, it creates a new QueryTermService, adds it to the queryTermServices array,
     * and updates the query terms list.
     *
     * @param queryValue - The value of the query term for which to generate a new QueryTermService.
     * @param searchResults - The number of search results to retrieve.
     * @param limitDistance - The maximum distance limit for neighbour terms.
     * @param graphTerms - The number of neighbour terms to include in the graph.
     */
    private generateNewQueryTermService(queryValue: string, searchResults: number, limitDistance: number, graphTerms: number): void {
        if (this.findQueryTermService(queryValue) === undefined) {
            let queryTermService = new QueryTermService(this, queryValue, searchResults, limitDistance, graphTerms)
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


class QueryTermsList {
    private readonly dynamicList: HTMLElement
    private readonly queryService: QueryService

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


class AddTermsTable {
    private activeTermService: QueryTermService | undefined
    private readonly dynamicTable: HTMLElement

    constructor() {
        this.dynamicTable = document.getElementById('addTermsTable') as HTMLElement
        const filterInput = document.getElementById('addTermsFilter') as HTMLInputElement;
        filterInput.addEventListener('input', () => this.filterTerms());
        this.toggleFilterVisibility();
    }

    public setActiveTermService(queryTermService: QueryTermService): void {
        this.activeTermService = queryTermService
        this.updateTable()
    }
    
    /**
    ​ * Updates the table with neighbour terms data.
    ​ * 
    ​ * This function retrieves the table body element, clears existing rows, and then iterates over the neighbour terms of the active query term.
    ​ * For each neighbour term, it creates a new row in the table and populates the cells with the term's value and hops.
    ​ * If the term is already in the visible neighbour terms list, it is not added to the table.
    ​ * 
    ​ * @remarks
    ​ * This function assumes that the table body element is already present in the HTML structure.
    ​ * 
    ​ * @returns {void}
    ​ */
    public updateTable(): void {
        // Get the table body element
        const tbody = this.dynamicTable.getElementsByTagName('tbody')[0]

        // Clear existing rows in the table
        tbody.innerHTML = '' 

        // Check if the activeTermService is defined
        if (this.activeTermService === undefined) return

        const visibleNeighbourTermsValues = this.activeTermService.getVisibleQueryTerm().getNeighbourTermsValues()

        // Iterate over the neighbour terms of the active query term
        for(const term of this.activeTermService.getCompleteQueryTerm().getNeighbourTerms()) {
            // Check if the term is not already in the visible neighbour terms list
            if ((!visibleNeighbourTermsValues.includes(term.getValue())) && (term.getCriteria() === "proximity")) {
                // Create a new row in the table
                const row = tbody.insertRow()

                // Create cells for the row
                const cell1 = row.insertCell(0)
                const cell2 = row.insertCell(1)

                // Set the text content of the first cell
                cell1.innerHTML = term.getValue()
                
                // Create the <i> element
                const icon = this.createIconElement(term)

                // Append the <i> element to the second cell
                cell2.appendChild(icon);
            }
        }

        // Toggle the filter input visibility, if the table has rows 
        this.toggleFilterVisibility()
    }

    /**
    ​ * Handles the addition of a neighbour term to the active query term's visible neighbour terms.
    ​ * 
    ​ * This function checks if the activeTermService is defined, retrieves the neighbour term by its value,
    ​ * and if the neighbour term exists, creates a new visible neighbour term and adds it to the active query term.
    ​ * 
    ​ * @param termValue - The value of the neighbour term to be added.
    ​ * 
    ​ * @returns {void}
    ​ */
    private handleTermAddition(termValue: string): void {
        if (this.activeTermService !== undefined) {
            const neighbourTerm = this.activeTermService.getCompleteQueryTerm().getNeighbourTermByValue(termValue)
            if (neighbourTerm !== undefined) {
                // Add the neighbour term to the active query term's visible neighbour terms
                const queryTerm = this.activeTermService.getVisibleQueryTerm()
                const value = neighbourTerm.getValue()
                const hops = neighbourTerm.getHops()
                const proximityPonderation = neighbourTerm.getProximityPonderation()
                const totalPonderation = neighbourTerm.getTotalPonderation()
                const criteria = neighbourTerm.getCriteria()
                const hopLimit = neighbourTerm.getHopLimit()

                let visibleNeighbourTerm = new NeighbourTerm(queryTerm, value, hops, proximityPonderation, totalPonderation, criteria, hopLimit)
                this.activeTermService.addVisibleNeighbourTerm(visibleNeighbourTerm)
            }
        }
    }

    /**
    ​ * Filters the terms in the 'addTermsTable' based on the input value in the 'addTermsFilter' input field.
    ​ * 
    ​ * This function retrieves the filter input element, the filter value, the table element, and the rows of the table.
    ​ * It then iterates over each row, retrieves the term cell, and checks if the term's lowercase value contains the filter value.
    ​ * If it does, the row's display style is set to '', making it visible. If it doesn't, the row's display style is set to 'none', making it hidden.
    ​ */
    private filterTerms(): void {
        const filterInput = document.getElementById('addTermsFilter') as HTMLInputElement;
        const filterValue = filterInput.value.toLowerCase();
        const table = document.getElementById('addTermsTable') as HTMLTableElement;
        const rows = Array.from(table.getElementsByTagName('tbody')[0].getElementsByTagName('tr'));
    
        for (const row of rows) {
            const termCell = row.getElementsByTagName('td')[0];
            const term = termCell.textContent ?? termCell.innerText;
    
            // Check if the term contains the filter value and set the row's display style accordingly
            if (term.toLowerCase().indexOf(filterValue) > -1) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        }
    }

    /**
     * Toggles the visibility of the addTermsFilter input based on the presence of rows in the addTermsTable.
     * 
     * This method checks whether there are any rows in the table's tbody. If there are no rows, the filter input
     * is hidden. If there are rows, the filter input is shown.
     */
    private toggleFilterVisibility(): void {
        // Get the table and its tbody element
        const table = document.getElementById('addTermsTable') as HTMLTableElement;
        const tbody = table.getElementsByTagName('tbody')[0];

        // Get the filter input
        const filterInput = document.getElementById('addTermsFilter') as HTMLInputElement;

        // Check if there are any rows in the tbody
        const rows = tbody.getElementsByTagName('tr');
        
        if (rows.length === 0) {
            // Hide the filter input if there are no rows
            filterInput.style.display = 'none';
        } else {
            // Show the filter input if there are rows
            filterInput.style.display = 'block';
        }
    }

    /**
    ​ * Creates an icon element for adding a neighbour term to the active query term's visible neighbour terms.
    ​ * 
    ​ * @param term - The neighbour term for which to create the icon element.
    ​ * 
    ​ * @returns A new HTML element representing the icon.
    ​ * 
    ​ * The icon is a fontawesome plus-circle icon with a pointer cursor.
    ​ * When clicked, it triggers the `handleTermAddition` method with the term's value as a parameter.
    ​ */
    private createIconElement(term: NeighbourTerm): HTMLElement {
        const icon = document.createElement('i');
        icon.className = 'fas fa-plus-circle';
        icon.style.cursor = 'pointer';

        // Add event listener to the icon element
        icon.addEventListener('click', () => {
            this.handleTermAddition(term.getValue());
        });

        return icon;
    }
}


class NeighbourTermsTable {
    private activeTermService: QueryTermService | undefined
    private readonly dynamicTable: HTMLElement

    constructor() {
        this.dynamicTable = document.getElementById('neighboursTermsTable') as HTMLElement
    }

    public setActiveTermService(queryTermService: QueryTermService): void {
        this.activeTermService = queryTermService
        this.updateTable()
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
            cell2.innerHTML = neighbourTerm.getCriteria()
        }
    }
}


class ResultsList {
    private activeTermService: QueryTermService | undefined
    private readonly dynamicList: HTMLElement

    constructor() {
        this.dynamicList = document.getElementById('resultsList') as HTMLElement
    }

    public setActiveTermService(queryTermService: QueryTermService): void {
        this.activeTermService = queryTermService
        this.updateList()
    }

    /**
     * Updates the list of query results with new documents.
     * Clears existing list items in the results list before adding new ones.
     * 
     * @remarks
     * This function is responsible for populating the results list with the documents retrieved from the active query term.
     * It iterates over the documents in the ranking and creates list items for each one, including title and abstract elements.
     * Click event listeners and mouse event listeners are added to the title elements to handle user interactions.
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
        const titleSentenceObject = [doc.getSentences()[0]];
        // Highlight the title element with green color for the query terms and purple color for the neighbour terms
        titleElement.appendChild(document.createTextNode((index + 1) + '. '));
        const highlightedSpanContainer = this.applyHighlightingToSentences(titleSentenceObject);
        titleElement.appendChild(highlightedSpanContainer);
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
        const abstractSentenceObjects = doc.getSentences().slice(1);
        // Highlight the abstract element with green color for the query terms and purple color for the neighbour terms
        const highlightedSpanContainer = this.applyHighlightingToSentences(abstractSentenceObjects);
        abstractElement.appendChild(highlightedSpanContainer);
        abstractElement.style.display = "none";
        return abstractElement;
    }

    /**
     * This function generates highlighted text for a given list of sentences.
     * It uses the query terms and neighbour terms to apply different colors to the words.
     *
     * @param sentenceObjects - An array of Sentence objects to generate highlighted text for.
     * @returns A span element containing the highlighted text for the given sentences.
     *
     * @remarks
     * The function retrieves the query terms and neighbour terms from the activeTermService,
     * and then applies highlighting to the words in the sentences based on these terms.
     * The highlighting is applied using different colors for the query terms and neighbour terms.
     */
    private applyHighlightingToSentences(sentenceObjects: Sentence[]): HTMLSpanElement {
        const queryTerms = this.activeTermService?.getVisibleQueryTerm().getValue() as string
        const queryTermsList = TextUtils.separateBooleanQuery(queryTerms)
        const neighbourTermsList = this.activeTermService?.getVisibleQueryTerm().getNeighbourTermsValues() as string[]
        return this.getHighlightedText(sentenceObjects, queryTermsList, neighbourTermsList);
    }

    /**
     * Applies highlighting to words in sentences based on query terms and neighbour terms.
     *
     * @param sentenceObjects - An array of Sentence objects to apply highlighting to.
     * @param queryTermsList - An array of strings representing the query terms.
     * @param neighbourTermsList - An array of strings representing the neighbour terms.
     *
     * @returns A span element containing the highlighted sentences.
     *
     * @remarks
     * This function iterates over each sentence object, splits the text into words,
     * and applies highlighting based on the presence of query terms and neighbour terms.
     * The function uses regular expressions to match the query terms and neighbour terms.
     * The highlighted words are wrapped in HTML span tags with a specific background color.
     * The function then joins the highlighted words back into sentences and returns the result.
     */
    private getHighlightedText(sentenceObjects: Sentence[], queryTermsList: string[], neighbourTermsList: string[]): HTMLSpanElement {
        if (sentenceObjects.length == 0) return document.createElement('span');
        let highlightedSentences: [string, Sentence | undefined][] = []

        for (let sentenceObject of sentenceObjects) {
            const sentenceText = sentenceObject.getRawText();
            if (sentenceObject.getQueryTerm().getNeighbourTerms().length == 0 || neighbourTermsList.length == 0) {
                // If there are no user neighbour terms in the sentence, just return the original sentence
                highlightedSentences.push([sentenceText, undefined]);
            } else {
                // If there are user neighbour terms in the sentence, split text by spaces and replace matching words
                const words = sentenceText.split(' ');
                const highlightedSentenceText = this.getHighlightedSentence(words, queryTermsList, neighbourTermsList);
                highlightedSentences.push([highlightedSentenceText, sentenceObject]);
            }
        }

        return this.applyEventListenersToSentences(highlightedSentences);
    }

    /**
     * This function generates a highlighted sentence based on the presence of query terms and neighbour terms.
     *
     * @param highlightedSentences - An array of strings representing the sentences to be highlighted.
     * @returns A HTMLSpanElement containing the highlighted sentences, with appropriate event listeners for mouseenter and mouseleave events.
     */
    private applyEventListenersToSentences(highlightedSentences: [string, Sentence | undefined][]): HTMLSpanElement {
        // Create the main container of type span
        const mainSpanContainer = document.createElement('span');

        // Add spans to the main container, and a separator in case it is not the last span
        highlightedSentences.forEach((sentence, index) => {
            const sentenceText = sentence[0];
            const sentenceObject = sentence[1];
            const spanElement = document.createElement('span');
            spanElement.innerHTML = sentenceText;

            // Add event listeners for mouseenter and mouseleave events if the sentence contains query terms or neighbour terms
            if (sentenceObject !== undefined) {
                spanElement.addEventListener("mouseenter", () => {
                    spanElement.style.backgroundColor = "#E4E4E4";
                    this.activeTermService?.setVisibleSentence(sentenceObject);
                });
            
                spanElement.addEventListener("mouseleave", () => {
                    spanElement.style.backgroundColor = "inherit";
                });
            }

            // Append the span element to the main container, with a separator if it is not the last span
            mainSpanContainer.appendChild(spanElement);
            if (index < (highlightedSentences.length - 1)) {
                mainSpanContainer.appendChild(document.createTextNode('. '));
            }
        });

        return mainSpanContainer;
    }

    /**
     * This function generates a highlighted sentence based on the presence of query terms and neighbour terms.
     *
     * @param words - An array of words that make up the sentence.
     * @param queryTermsList - A list of query terms to be checked against.
     * @param neighbourTermsList - A list of neighbour terms to be checked against.
     *
     * @returns A string representing the highlighted sentence. Each query term and neighbour term is highlighted
     * with a different background color. Non-matching words are returned as is.
     */
    private getHighlightedSentence(words: string[], queryTermsList: string[], neighbourTermsList: string[]): string {
        const highlightedSentence = words.map((word, index) => {
            // Recreate regex objects in each iteration to avoid state issues with global regex
            const queryTermsRegex = new RegExp(queryTermsList.join('|'), 'gi');
            const neighbourTermsRegex = new RegExp(neighbourTermsList.join('|'), 'gi');

            if (neighbourTermsRegex.test(word)) {
                return this.getHighlightedWordIfNeighbourTerm(word, words, index, queryTermsList);
            } else if (queryTermsRegex.test(word)) {
                return `<span style="background-color: #98EE98;">${word}</span>`;
            } else {
                return word;
            }
        }).join(' ');

        return highlightedSentence;
    }

    /**
     * This function checks if a given word is a query term within a specified hop limit and 
     * returns the word with a highlighted background color if it is a query term.
     *
     * @param word - The word to be checked for being a query term.
     * @param words - An array of words surrounding the given word.
     * @param index - The index of the given word in the words array.
     * @param queryTermsList - A list of query terms to be checked against.
     *
     * @returns A string representing the given word with a highlighted background color if 
     * it is a query term. If it is not a query term, the original word is returned.
     */
    private getHighlightedWordIfNeighbourTerm(word: string, words: string[], index: number, queryTermsList: string[]): string {
        const stopwords = ["a", "about", "above", "accordingly", "after", "against", "ain", "all", "also", "although", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "besides", "between", "both", "but", "by", "can", "can't", "cannot", "consequently", "could", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "due", "during", "each", "etc", "every", "few", "for", "from", "further", "furthermore", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "however", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "let's", "likewise", "ll", "m", "ma", "me", "might", "mightn", "mightn't", "more", "moreover", "most", "must", "mustn", "mustn't", "my", "myself", "needn", "needn't", "nevertheless", "no", "nonetheless", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she'd", "she'll", "she's", "should", "should've", "shouldn", "shouldn't", "similarly", "since", "so", "some", "such", "t", "than", "that", "that'll", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "therefore", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "though", "through", "thus", "to", "too", "under", "unless", "until", "up", "using", "ve", "very", "was", "wasn", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren", "weren't", "what", "what's", "when", "when's", "where", "where's", "whereas", "whether", "which", "while", "who", "who's", "whom", "whose", "why", "why's", "will", "with", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        const hopLimit = this.activeTermService?.getVisibleQueryTerm().getNeighbourTerms()[0].getHopLimit() ?? 0
        let foundQueryTerm = false;

        // Check words to the left
        if (index > 0) {
            foundQueryTerm = this.checkQueryTermToTheLeft(words, index, hopLimit, stopwords, queryTermsList)
        }

        // Check words to the right
        if (index < words.length - 1 && !foundQueryTerm) {
            foundQueryTerm = this.checkQueryTermToTheRight(words, index, hopLimit, stopwords, queryTermsList)
        }

        if (foundQueryTerm) {
            return `<span style="background-color: #D8D8EE;">${word}</span>`;
        } else {
            return word;
        }
    }

    /**
     * This function checks if a given word is a query term within a specified hop limit to the left of the given index.
     *
     * @param words - An array of words surrounding the given word.
     * @param index - The index of the given word in the words array.
     * @param hopLimit - The maximum number of words to the left of the given index to check for query terms.
     * @param stopwords - An array of stopwords to be ignored when checking for query terms.
     * @param queryTermsList - A list of query terms to be checked against.
     *
     * @returns A boolean indicating whether a query term is found within the specified hop limit to the left of the given index.
     */
    private checkQueryTermToTheLeft(words: string[], index: number, hopLimit: number, 
        stopwords: string[], queryTermsList: string[]): boolean {
        return this.checkQueryTerm(words, index, hopLimit, stopwords, queryTermsList, -1);
    }

    /**
     * This function checks if a given word is a query term within a specified hop limit to the right of the given index.
     *
     * @param words - An array of words surrounding the given word.
     * @param index - The index of the given word in the words array.
     * @param hopLimit - The maximum number of words to the right of the given index to check for query terms.
     * @param stopwords - An array of stopwords to be ignored when checking for query terms.
     * @param queryTermsList - A list of query terms to be checked against.
     *
     * @returns A boolean indicating whether a query term is found within the specified hop limit to the right of the given index.
     */
    private checkQueryTermToTheRight(words: string[], index: number, hopLimit: number, 
        stopwords: string[], queryTermsList: string[]): boolean {
        return this.checkQueryTerm(words, index, hopLimit, stopwords, queryTermsList, 1);
    }

    /**
     * This helper function checks if a query term is found within a specified hop limit in a given direction.
     *
     * @param words - An array of words surrounding the given word.
     * @param startIndex - The starting index to begin checking from.
     * @param hopLimit - The maximum number of words to check for query terms.
     * @param stopwords - An array of stopwords to be ignored when checking for query terms.
     * @param queryTermsList - A list of query terms to be checked against.
     * @param direction - The direction to check in, either 1 (right) or -1 (left).
     *
     * @returns A boolean indicating whether a query term is found within the specified hop limit in the given direction.
     */
    private checkQueryTerm(words: string[], startIndex: number, hopLimit: number, stopwords: string[], 
        queryTermsList: string[], direction: 1 | -1): boolean {
        let counter = 0;
        const increment = direction === 1 ? 1 : -1;
        
        for (let i = startIndex + increment; i >= 0 && i < words.length && counter < hopLimit; i += increment) {
            const word = words[i];
            if (!stopwords.includes(word.toLowerCase())) {
                const queryTermsRegex = new RegExp(queryTermsList.join('|'), 'gi');
                if (queryTermsRegex.test(word)) {
                    return true;
                }
                if (word.includes('-')) {
                    const splitWords = word.split('-');
                    const stopWordsCount = splitWords.filter(w => stopwords.includes(w.toLowerCase())).length;
                    counter += (splitWords.length - 1) - stopWordsCount;
                }
                counter++;
            }
        }
        return false;
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


/**
 * Represents a component responsible for handling query input interactions.
 * 
 * This class is responsible for capturing user input from an HTML input element,
 * and sending the query to a query service when the Enter key is pressed.
 */
class QueryComponent {
    private readonly queryService: QueryService
    private readonly input: HTMLInputElement
    private readonly searchIcon: HTMLElement
    private readonly searchResultsInput: HTMLInputElement
    private readonly limitDistanceInput: HTMLInputElement
    private readonly graphTermsInput: HTMLInputElement

    /**
     * Constructs a new instance of QueryComponent.
     * This class is responsible for handling query input interactions.
     *
     * @param queryService - The QueryService instance to be used for sending queries.
     *
     * @remarks
     * The QueryComponent captures user input from an HTML input element,
     * and sends the query to the QueryService when the Enter key is pressed or the search icon is clicked.
     */
    constructor(queryService: QueryService) {
        this.queryService = queryService
        this.input = document.getElementById('queryInput') as HTMLInputElement
        this.searchIcon = document.getElementById('searchIcon') as HTMLElement
        this.searchResultsInput = document.getElementById('searchResults') as HTMLInputElement;
        this.limitDistanceInput = document.getElementById('limitDistance') as HTMLInputElement;
        this.graphTermsInput = document.getElementById('graphTerms') as HTMLInputElement;

        // Set default values for the inputs
        this.searchResultsInput.value = "10";
        this.limitDistanceInput.value = "4";
        this.graphTermsInput.value = "5";

        this.addEventListeners()
    }

    /**
     * Handles the query input and adds event listeners to various elements.
     * 
     * This function adds event listeners for:
     * - "Enter" key presses in the query input field.
     * - Clicking the search icon.
     * - Toggling the visibility of search parameters.
     * - Validating and ensuring inputs stay within defined ranges.
     */
    private addEventListeners(): void {
        // Event listener for "Enter" key presses
        this.input.addEventListener("keyup", event => {
            if(event.key === "Enter") {
                this.processQuery()
            }
        })

        // Event listener for clicking the search icon
        this.searchIcon.addEventListener("click", () => {
            this.processQuery()
        })

         // Add validation to ensure inputs stay within defined ranges
         // Validate search results input
         this.searchResultsInput.addEventListener("change", () => {
            let value = parseInt(this.searchResultsInput.value, 10);
            if (isNaN(value) || value < 5) {
                this.searchResultsInput.value = "5";
            }
        });

        // Validate limit distance input
        this.limitDistanceInput.addEventListener("change", () => {
            let value = parseInt(this.limitDistanceInput.value, 10);
            if (isNaN(value) || value < 2) {
                this.limitDistanceInput.value = "2";
            } else if (value > 10) {
                this.limitDistanceInput.value = "10";
            }
        });

        // Validate number of graph terms input
        this.graphTermsInput.addEventListener("change", () => {
            let value = parseInt(this.graphTermsInput.value, 10);
            if (isNaN(value) || value < 1) {
                this.graphTermsInput.value = "1";
            } else if (value > 20) {
                this.graphTermsInput.value = "20";
            }
        });
    }

    /**
     * Handles the query input and sends the query to the query service.
     * 
     * This function retrieves the trimmed input value from the query input field, clears the input field,
     * and checks if the value contains at least one alphanumeric character. If it does, it sends the query to the query service.
     * If the value is empty or does not contain any alphanumeric characters, it alerts the user to enter a valid query.
     */
    private processQuery() {
        let queryValue = this.input.value.trim() // Get the trimmed input value
        this.input.value = '' // Clear the input field
        const alphanumericRegex = /[a-zA-Z0-9]/
        if (alphanumericRegex.test(queryValue)) {   // Check if the value contains at least one alphanumeric character
            const searchResults = parseInt(this.searchResultsInput.value, 10);
            const limitDistance = parseInt(this.limitDistanceInput.value, 10);
            const graphTerms = parseInt(this.graphTermsInput.value, 10);
            this.queryService.setQuery(queryValue, searchResults, limitDistance, graphTerms) // Send the query to the query service
        } else if (queryValue !== '') {
            alert("Please enter a valid query.")    // Alert the user if the query is invalid
        }
    }
}


class RerankComponent {
    private readonly queryService: QueryService
    private readonly button: HTMLButtonElement

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
            console.log(ranking)
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





const cyUser = cytoscape({
    container: document.getElementById("cy") as HTMLElement,
    layout: {
        name: "preset",
    },
    style: [
        {
            selector: '.' + NodeType.central_node,
            style: {
            "background-color": '#ff8000',
            'width': '20px',
            'height': '20px',
            'label': "data(id)",
            'font-size': '13px',
            'color': '#5d5d5d'
            },
        },
        {
            selector: "edge",
            style: {
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "line-color": "#ccc",
            "width": "2px", // set the width of the edge
            "font-size": "12px" // set the font size of the label            
            },
        },
        {
            selector: '.' + NodeType.outer_node,
            style: {
              'background-color': '#8080EE',
              'width': '15px',
              'height': '15px',
              'label': 'data(label)',
              'font-size': '12px',
              'color': '#4b4b4b'
            }
        }
    ],
    userZoomingEnabled: false,
    userPanningEnabled: false
})


const cySentence = cytoscape({
    container: document.getElementById("cySentence") as HTMLElement,
    layout: {
        name: "preset",
    },
    style: [
        {
            selector: '.' + NodeType.central_node,
            style: {
            "background-color": '#70CC70',
            'width': '16px',
            'height': '16px',
            'label': "data(id)",
            'font-size': '10px',
            'color': '#5d5d5d'
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
            "font-size": "11px" // set the font size of the label            
            },
        },
        {
            selector: '.' + NodeType.outer_node,
            style: {
                'background-color': '#A0A0EE',
                'width': '12px',
                'height': '12px',
                'label': 'data(label)',
                'font-size': '9px',
                'color': '#4b4b4b'
            }
        }
    ],
    userZoomingEnabled: false,
    userPanningEnabled: false
})

// When the user drags a node
cyUser.on('drag', 'node', evt => {
    queryService.getActiveQueryTermService()?.nodeDragged(evt.target.id(), evt.target.position())
})

// When the user right-clicks a node
cyUser.on('cxttap', "node", evt => {
    queryService.getActiveQueryTermService()?.removeVisibleNeighbourTerm(evt.target.id())
});

// When the user right-clicks a edge
cyUser.on('cxttap', "edge", evt => {
    queryService.getActiveQueryTermService()?.removeVisibleNeighbourTerm(evt.target.id().substring(2))
});

// When the user hovers it's mouse over a node
cyUser.on('mouseover', 'node', (evt: cytoscape.EventObject) => {
    queryService.getActiveQueryTermService()?.changeCursorType(evt.target.id(), 'pointer');
});

// When the user moves it's mouse away from a node
cyUser.on('mouseout', 'node', (evt: cytoscape.EventObject) => {
    queryService.getActiveQueryTermService()?.changeCursorType(evt.target.id(), 'default');
});



const queryService: QueryService = new QueryService()
const queryComponent: QueryComponent = new QueryComponent(queryService)
const rerankComponent: RerankComponent = new RerankComponent(queryService)


cyUser.ready(() => {})


// quick way to get instances in console
;(window as any).cy = cyUser
;(window as any).queryService = queryService
